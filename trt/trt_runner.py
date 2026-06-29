"""Minimal TensorRT runtime helper for GridTrackNet.

Loads a serialized engine and runs inference on a contiguous float32 array of
shape (N, 15, 432, 768), returning float32 (N, 15, 27, 48). Device buffers are
allocated lazily and reused (grown on demand), so repeated calls are cheap.
"""
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

_HtoD = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
_DtoH = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


def _check(ret):
    """cuda-python calls return (err, *results); raise on error, return results."""
    err, *rest = ret if isinstance(ret, (tuple, list)) else (ret,)
    if err != cudart.cudaError_t.cudaSuccess:
        name = cudart.cudaGetErrorString(err)[1].decode()
        raise RuntimeError(f"CUDA error: {name}")
    return rest[0] if len(rest) == 1 else tuple(rest)


class TRTRunner:
    def __init__(self, engine_path):
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self.engine = self._runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        # Largest batch the engine's optimization profile supports.
        self.max_batch = int(self.engine.get_tensor_profile_shape(self.input_name, 0)[2][0])

        self.stream = _check(cudart.cudaStreamCreate())
        self._d_in = self._d_out = None
        self._in_cap = self._out_cap = 0

    def _device_buf(self, ptr, cap, nbytes):
        if nbytes > cap:
            if ptr is not None:
                _check(cudart.cudaFree(ptr))
            ptr = _check(cudart.cudaMalloc(nbytes))
            cap = nbytes
        return ptr, cap

    def infer(self, x):
        """Run inference on (N, 15, 432, 768). Batches larger than the engine's
        max profile are split into chunks and concatenated, so any N is valid."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape[0] > self.max_batch:
            return np.concatenate(
                [self._infer_one(x[i:i + self.max_batch]) for i in range(0, x.shape[0], self.max_batch)],
                axis=0,
            )
        return self._infer_one(x)

    def _infer_one(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.context.set_input_shape(self.input_name, x.shape)
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out = np.empty(out_shape, dtype=np.float32)

        self._d_in, self._in_cap = self._device_buf(self._d_in, self._in_cap, x.nbytes)
        self._d_out, self._out_cap = self._device_buf(self._d_out, self._out_cap, out.nbytes)

        _check(cudart.cudaMemcpyAsync(self._d_in, x.ctypes.data, x.nbytes, _HtoD, self.stream))
        self.context.set_tensor_address(self.input_name, int(self._d_in))
        self.context.set_tensor_address(self.output_name, int(self._d_out))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")
        _check(cudart.cudaMemcpyAsync(out.ctypes.data, self._d_out, out.nbytes, _DtoH, self.stream))
        _check(cudart.cudaStreamSynchronize(self.stream))
        return out

    def infer_device(self, dev_ptr, shape):
        """Zero-copy inference from an already-on-GPU input. dev_ptr is a CUDA device
        pointer (int) to a contiguous float32 (N, 15, 432, 768) buffer; the caller must
        ensure it's fully produced (synchronized) before calling. Only the small output
        grid is copied back to host. Requires N <= max_batch."""
        n = int(shape[0])
        if n > self.max_batch:
            raise ValueError(f"batch {n} exceeds engine max_batch {self.max_batch}")
        self.context.set_input_shape(self.input_name, tuple(int(s) for s in shape))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out = np.empty(out_shape, dtype=np.float32)
        self._d_out, self._out_cap = self._device_buf(self._d_out, self._out_cap, out.nbytes)

        self.context.set_tensor_address(self.input_name, int(dev_ptr))
        self.context.set_tensor_address(self.output_name, int(self._d_out))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")
        _check(cudart.cudaMemcpyAsync(out.ctypes.data, self._d_out, out.nbytes, _DtoH, self.stream))
        _check(cudart.cudaStreamSynchronize(self.stream))
        return out

    def close(self):
        for ptr in (self._d_in, self._d_out):
            if ptr is not None:
                cudart.cudaFree(ptr)
        self._d_in = self._d_out = None
        if self.stream is not None:
            cudart.cudaStreamDestroy(self.stream)
            self.stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
