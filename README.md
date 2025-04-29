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


## Setup

### Create virtual environment
uv venv

### Activate the virtual environment
#### On Windows:
.venv\Scripts\activate
#### On macOS/Linux:
source .venv/bin/activate

# Install dependencies using uv sync
uv sync


## Running


- Debug: uv run main.py --train --debug
- Debug w/ transfer learning: uv run main.py --train --debug --load
- Inference: uv run main.py
