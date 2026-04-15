
import os
import sys
import pathlib

# Copy the get_mlx_paths logic from setup.py
def get_mlx_paths():
    """Locate mlx include and lib directories."""
    # Strategy 1: Try to find via mlx.core import
    try:
        import mlx.core
        mlx_path = pathlib.Path(mlx.core.__file__).parent
        inc = str(mlx_path / "include")
        lib = str(mlx_path / "lib")
        print(f"Checking Strategy 1: {inc}")
        if os.path.exists(os.path.join(inc, "mlx/mlx.h")):
            return inc, lib
    except Exception as e:
        print(f"Strategy 1 failed: {e}")

    # Strategy 2: Search in sys.path
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
        
        # Also check local .venv if it exists
        cwd = os.getcwd()
        for venv in [".venv", "venv", "myvnv"]:
            for lib_dir in ["lib", "lib64"]:
                venv_lib = os.path.join(cwd, venv, lib_dir)
                if os.path.exists(venv_lib):
                    for pyver in os.listdir(venv_lib):
                        if pyver.startswith("python"):
                            site_pkg = os.path.join(venv_lib, pyver, "site-packages")
                            search_paths.append(site_pkg)

        for sp in search_paths:
            if not os.path.exists(sp): continue
            inc = os.path.join(sp, "mlx/include")
            print(f"Checking SP: {inc}")
            if os.path.exists(os.path.join(inc, "mlx/mlx.h")):
                lib = os.path.join(sp, "mlx/lib")
                return inc, lib
    except Exception as e:
        print(f"Strategy 3 failed: {e}")

    # Strategy 4: Local source fallback
    local_mlx = os.path.abspath(os.path.join(os.getcwd(), "..", "mlx"))
    if os.path.exists(os.path.join(local_mlx, "mlx/mlx.h")):
        return local_mlx, os.path.join(local_mlx, "build")

    return None, None

inc, lib = get_mlx_paths()
print(f"RESULT: inc={inc}, lib={lib}")
