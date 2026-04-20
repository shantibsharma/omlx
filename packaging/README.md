# cMLX macOS App Packaging

Packages cMLX as a macOS menubar app using venvstacks.

## Requirements

- macOS 15.0+ (Sequoia) — required by MLX >= 0.29.2
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- venvstacks: `pip install venvstacks`

## Build

```bash
cd packaging

# Full build (venvstacks + app bundle + DMG)
python build.py

# Skip venvstacks build (use existing environment)
python build.py --skip-venv

# DMG only (use existing build)
python build.py --dmg-only
```

## Output

```
packaging/
├── build/
│   ├── venvstacks/     # venvstacks build cache
│   ├── envs/           # Exported environments
│   └── cMLX.app/       # App bundle
└── dist/
    └── cMLX-<version>.dmg  # Distribution DMG
```

## Structure

```
cMLX.app/
├── Contents/
│   ├── Info.plist
│   ├── MacOS/
│   │   └── cMLX           # Launcher script
│   ├── Resources/
│   │   ├── cmlx_app/      # Menubar app
│   │   ├── cmlx/          # cMLX server
│   │   └── AppIcon.icns
│   └── Frameworks/
│       ├── cpython3.11/   # Python runtime
│       ├── mlx-framework/ # MLX + dependencies
│       └── cmlx-app/      # App layer
```

## Layer Configuration

| Layer | Contents |
|-------|----------|
| Runtime | Python 3.11 |
| Framework | MLX, mlx-lm, mlx-vlm, FastAPI, transformers |
| Application | rumps, PyObjC |

## Installation

1. Open the DMG file
2. Drag cMLX.app to the Applications folder
3. Launch the app (appears in the menubar)
4. Set the model directory in Settings
5. Click Start Server
