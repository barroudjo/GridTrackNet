"""Locating a usable ffmpeg/ffprobe binary.

The system PATH may not contain ffmpeg, so a few known install locations are
probed as well. Some builds need their own lib directory on LD_LIBRARY_PATH.
"""
import os
import shutil
import subprocess

_CANDIDATES = [
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/opt/AgentDVR/ffmpeg7/bin/ffmpeg",
]


def _env_for(binary):
    env = os.environ.copy()
    libdir = os.path.join(os.path.dirname(os.path.dirname(binary)), "lib")
    if os.path.isdir(libdir):
        env["LD_LIBRARY_PATH"] = libdir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _works(binary):
    if not (binary and os.path.exists(binary)):
        return False
    try:
        subprocess.run([binary, "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, env=_env_for(binary), check=True)
        return True
    except Exception:
        return False


def find_ffmpeg(name="ffmpeg"):
    """Returns (path, env) for ffmpeg or ffprobe, or (None, None) if unavailable."""
    paths = [shutil.which(name)] + [c.replace("ffmpeg", name) for c in _CANDIDATES]
    for p in paths:
        if _works(p):
            return p, _env_for(p)
    return None, None


def run(name, args, **kwargs):
    binary, env = find_ffmpeg(name)
    if binary is None:
        raise RuntimeError("%s not found" % name)
    return subprocess.run([binary] + args, env=env, **kwargs)
