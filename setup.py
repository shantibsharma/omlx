from setuptools import setup, Extension
import os
import pathlib

def get_mlx_paths():
    """Locate mlx include and lib directories."""
    try:
        import mlx.core
        mlx_path = pathlib.Path(mlx.core.__file__).parent
        return str(mlx_path / "include"), str(mlx_path / "lib")
    except Exception:
        return None, None

extensions = [
    Extension(
        "omlx.omlx_fast_io",
        sources=[
            "src/omlx_fast_io_wrapper.cpp",
            "src/cache_core.cpp",
            "src/scheduler_core.cpp",
            "src/omlx_fast_io.cpp",
        ],
        include_dirs=["src"],
        libraries=["mlx"],
        extra_compile_args=["-O3", "-std=c++17", "-fPIC"],
    )
]

# We use a custom build command to dynamically inject the mlx paths
# just before compilation starts, after dependencies are guaranteed to be installed.
from setuptools.command.build_ext import build_ext

class BuildExtWithMLX(build_ext):
    def run(self):
        include_dirs, library_dirs = get_mlx_paths()
        if include_dirs and library_dirs:
            for ext in self.extensions:
                ext.include_dirs.append(include_dirs)
                ext.library_dirs.append(library_dirs)
                ext.extra_link_args.append(f"-Wl,-rpath,{library_dirs}")
        super().run()

setup(
    name="omlx",
    version="0.1.0",
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtWithMLX},
)
