from setuptools import setup, Extension
import os
import pathlib
import sys

def get_mlx_paths():
    """Locate mlx include and lib directories."""
    # Strategy 1: Try to find via mlx.core import
    try:
        import mlx.core
        mlx_path = pathlib.Path(mlx.core.__file__).parent
        inc = str(mlx_path / "include")
        lib = str(mlx_path / "lib")
        if os.path.exists(os.path.join(inc, "mlx/mlx.h")):
            return inc, lib
    except Exception:
        pass

    # Strategy 2: Search in sys.path (useful if installed in a venv that's active)
    for p in sys.path:
        if not p: continue
        inc = os.path.join(p, "mlx/include")
        if os.path.exists(os.path.join(inc, "mlx/mlx.h")):
            lib = os.path.join(p, "mlx/lib")
            return inc, lib

    # Strategy 3: Try to find via site-packages search
    try:
        import site
        search_paths = site.getsitepackages()
        if hasattr(site, 'getusersitepackages'):
            search_paths.append(site.getusersitepackages())
        
        # Also check local .venv if it exists relative to this setup.py
        this_dir = os.path.dirname(os.path.abspath(__file__))
        for venv in [".venv", "venv", "myvnv", "../../build/omlx/myvnv"]:
            # Check absolute or relative to setup.py
            venv_candidates = [
                os.path.join(this_dir, venv),
                os.path.abspath(os.path.join(this_dir, "..", "..", "build", "omlx", "myvnv"))
            ]
            for venv_path in venv_candidates:
                if not os.path.exists(venv_path): continue
                for lib_dir in ["lib", "lib64"]:
                    venv_lib = os.path.join(venv_path, lib_dir)
                    if os.path.exists(venv_lib):
                        for pyver in os.listdir(venv_lib):
                            if pyver.startswith("python"):
                                site_pkg = os.path.join(venv_lib, pyver, "site-packages")
                                search_paths.append(site_pkg)

        for sp in search_paths:
            if not os.path.exists(sp): continue
            inc = os.path.join(sp, "mlx/include")
            # print(f"🔍 Checking SP: {inc}") # Debug
            if os.path.exists(os.path.join(inc, "mlx/mlx.h")):
                lib = os.path.join(sp, "mlx/lib")
                return inc, lib
    except Exception:
        pass

    # Strategy 4: Local source fallback (last resort)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    local_mlx = os.path.abspath(os.path.join(this_dir, "..", "mlx"))
    if os.path.exists(os.path.join(local_mlx, "mlx/mlx.h")):
        return local_mlx, os.path.join(local_mlx, "build")

    return None, None

extensions = [
    Extension(
        "omlx.omlx_fast_io",
        sources=[
            "src/omlx_fast_io_wrapper.cpp",
            "src/cache_core.cpp",
            "src/scheduler_core.cpp",
            "src/native_engine.cpp",
            "src/native_ssd_cache.cpp",
            "src/llama_model.cpp",
            "src/metal_ops.cpp",
            "src/omlx_fast_io.cpp",
            "src/paged_attention.cpp",
        ],
        include_dirs=["src", "src/metal"],
        libraries=["mlx"],
        extra_compile_args=["-O3", "-std=c++17", "-fPIC"],
    )
]

from setuptools.command.build_ext import build_ext

class BuildExtWithMLX(build_ext):
    def run(self):
        print("🔍 Searching for MLX headers and libraries...")
        include_dirs, library_dirs = get_mlx_paths()

        if include_dirs:
            print(f"✅ Found MLX! Include: {include_dirs}, Lib: {library_dirs}")
            for ext in self.extensions:
                ext.include_dirs.append(include_dirs)
                # Add metal_cpp for <Metal/Metal.hpp>
                ext.include_dirs.append(os.path.join(include_dirs, "metal_cpp"))
                ext.library_dirs.append(library_dirs)
                ext.extra_link_args.append(f"-Wl,-rpath,{library_dirs}")
        else:
            print("❌ ERROR: Could not find MLX headers. Make sure 'pip install mlx' was successful.")
            # We don't exit here so the user sees the error message from the build process

        super().run()

setup(
    name="omlx",
    version="0.1.0",
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtWithMLX},
)
