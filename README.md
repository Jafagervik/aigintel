# GINTEL AI WORKSHOP

## Setup 

### Windows

```terminal
scripts\install.bat
```

### Mac

```shell
chmod +x scripts/install.sh && scripts/install.sh
```

Install Ruff plugin in Pycharm afterwards


## Running


- Debug: uv run main.py --train --debug
- Debug w/ transfer learning: uv run main.py --train --debug --load
- Inference: uv run main.py
