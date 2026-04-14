from setuptools import setup, Extension
import os
import pathlib

def get_mlx_paths():
    """Locate mlx include and lib directories."""
    try:
        import mlx.core
        mlx_path = pathlib.Path(mlx.core.__file__).parent
        return str(mlx_path / "include"), str(mlx_path / "lib")
    except Exception as e:
        print(f"Error locating mlx paths: {e}")
        return None, None

include_dirs, library_dirs = get_mlx_paths()

if not include_dirs or not library_dirs:
    include_dirs = []
    library_dirs = []

extensions = [
    Extension(
        "omlx.omlx_fast_io",
        sources=[
            "src/omlx_fast_io_wrapper.cpp",
            "src/cache_core.cpp",
            "src/scheduler_core.cpp",
            "src/omlx_fast_io.cpp",
        ],
        include_dirs=[include_dirs, "src"] if include_dirs else ["src"],
        library_dirs=[library_dirs] if library_dirs else [],
        libraries=["mlx"],
        extra_compile_args=["-O3", "-std=c++17", "-fPIC"],
        extra_link_args=["-Wl,-rpath," + library_dirs] if library_dirs else [],
    )
]

setup(
    name="omlx",
    version="0.1.0",
    ext_modules=extensions,
)
