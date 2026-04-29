# Hailo Dataflow Compiler + Model Zoo wheels (workstation only)

Local cache of the EULA-gated wheels needed to compile YOLO `.pt`
weights down to `.hef` for the Pi. **Never committed** — the wheels
are gitignored (`resources/hailo-sdk/*.whl`) because each developer
pulls their own copy under their Hailo Developer Zone entitlement.

If this directory is empty when you check out the repo, that is
expected. Download from <https://hailo.ai/developer-zone/> (free
account, EULA acceptance) and drop the four files here:

```
hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl    (~489 MB)  -> AI HAT+   (Hailo-8 / 8L)
hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl     (~499 MB)  -> AI HAT+ 2 (Hailo-10H)
hailo_model_zoo-2.18.0-py3-none-any.whl                     (~750 KB)  -> AI HAT+   (paired with DFC 3.33.1)
hailo_model_zoo-5.3.0-py3-none-any.whl                      (~890 KB)  -> AI HAT+ 2 (paired with DFC 5.3.0)
```

The DFCs from Hailo's 4.x line (DFC 3.33.x) and 5.x line (DFC 5.3.x)
share the top-level `hailo_sdk_client` package, so they cannot
coexist in the same Python environment. Build one venv per arch.

## Two-venv setup

Hailo's wheels target Python 3.10 — install via `uv` if your system
Python differs:

```bash
uv python install 3.10
ln -sf ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 ~/.local/bin/
```

### AI HAT+ compile venv (Hailo-8 / 8L)

```bash
python3.10 -m venv venv-compile-h8
./venv-compile-h8/bin/pip install --upgrade pip wheel setuptools
./venv-compile-h8/bin/pip install -e ".[dev]"
./venv-compile-h8/bin/pip install resources/hailo-sdk/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
./venv-compile-h8/bin/pip install resources/hailo-sdk/hailo_model_zoo-2.18.0-py3-none-any.whl
./venv-compile-h8/bin/python -c "import hailo_sdk_client; print(hailo_sdk_client.__version__)"

BENCHY_VENV=venv-compile-h8 scripts/compile_workstation_hefs.sh --arch hailo8
```

### AI HAT+ 2 compile venv (Hailo-10H)

```bash
python3.10 -m venv venv-compile-h10h
./venv-compile-h10h/bin/pip install --upgrade pip wheel setuptools
./venv-compile-h10h/bin/pip install -e ".[dev]"
./venv-compile-h10h/bin/pip install resources/hailo-sdk/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl
./venv-compile-h10h/bin/pip install resources/hailo-sdk/hailo_model_zoo-5.3.0-py3-none-any.whl
./venv-compile-h10h/bin/python -c "import hailo_sdk_client; print(hailo_sdk_client.__version__)"

BENCHY_VENV=venv-compile-h10h scripts/compile_workstation_hefs.sh --arch hailo10h --include-detection
```

## Why two venvs

The 4.x and 5.x DFC lines are not API-compatible and ship the same
`hailo_sdk_client` top-level package. Installing both into one venv
causes the second pip install to clobber the first; you'd end up
compiling everything for whichever line you installed last and
producing HEFs the other Pi can't load. One venv per arch is the
guaranteed-correct setup.

The Pi-side runtime is a separate matter — `resources/hailo-8/`
ships HailoRT 4.x `.deb` installers for the AI HAT+ Pi and
`resources/hailo-10H/` ships HailoRT 5.x for the AI HAT+ 2 Pi.
Those go on the Pi, not here.
