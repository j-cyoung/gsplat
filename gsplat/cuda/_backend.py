import glob
import json
import os
import shutil
from subprocess import DEVNULL, call

from rich.console import Console
from torch.utils.cpp_extension import _get_build_directory, load

PATH = os.path.dirname(os.path.abspath(__file__))


def _env_flag(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return 1
    if value in ("0", "false", "no", "off"):
        return 0
    raise ValueError(
        f"Invalid value for {name}: {value!r}. Expected one of 0/1/true/false/on/off."
    )


def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def cuda_toolkit_version():
    """Get the cuda toolkit version."""
    cuda_home = os.path.join(os.path.dirname(shutil.which("nvcc")), "..")
    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the cuda version.")
    return cuda_version


name = "gsplat_cuda"
build_dir = _get_build_directory(name, verbose=False)
extra_include_paths = [os.path.join(PATH, "csrc/third_party/glm")]
ENABLE_PREFILTER = _env_flag("GSPLAT_ENABLE_PREFILTER", 1)
ENABLE_SSAA = _env_flag("GSPLAT_ENABLE_SSAA", 1)
extra_cflags = [
    "-O3",
    f"-DGSPLAT_ENABLE_PREFILTER={ENABLE_PREFILTER}",
    f"-DGSPLAT_ENABLE_SSAA={ENABLE_SSAA}",
]
extra_cuda_cflags = [
    "-O3",
    f"-DGSPLAT_ENABLE_PREFILTER={ENABLE_PREFILTER}",
    f"-DGSPLAT_ENABLE_SSAA={ENABLE_SSAA}",
]

_C = None
sources = list(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + list(
    glob.glob(os.path.join(PATH, "csrc/*.cpp"))
)
# sources = [
#     os.path.join(PATH, "csrc/ext.cpp"),
#     os.path.join(PATH, "csrc/rasterize.cu"),
#     os.path.join(PATH, "csrc/bindings.cu"),
#     os.path.join(PATH, "csrc/forward.cu"),
#     os.path.join(PATH, "csrc/backward.cu"),
# ]

try:
    # try to import the compiled module (via setup.py)
    from gsplat import csrc as _C
except ImportError:
    # if failed, try with JIT compilation
    if cuda_toolkit_available():
        # If JIT is interrupted it might leave a lock in the build directory.
        # We dont want it to exist in any case.
        try:
            os.remove(os.path.join(build_dir, "lock"))
        except OSError:
            pass

        if os.path.exists(os.path.join(build_dir, "gsplat_cuda.so")) or os.path.exists(
            os.path.join(build_dir, "gsplat_cuda.lib")
        ):
            # If the build exists, we assume the extension has been built
            # and we can load it.

            _C = load(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
            )
        else:
            # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
            # if the build directory exists with a lock file in it.
            shutil.rmtree(build_dir)
            with Console().status(
                "[bold yellow]gsplat: Setting up CUDA (This may take a few minutes the first time)",
                spinner="bouncingBall",
            ):
                _C = load(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                )
    else:
        Console().print(
            "[yellow]gsplat: No CUDA toolkit found. gsplat will be disabled.[/yellow]"
        )


__all__ = ["_C"]
