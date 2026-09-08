# VoiceDenoiser

Two things live in this workspace:

| Folder | What it is |
|---|---|
| **`AudioDenoiser\`** | The app we built: real-time microphone denoising with a UI (Live tab) + batch file denoising (Files tab), powered by DeepFilterNet3. **Start here.** See its own [README](AudioDenoiser/README.md) for controls, tuning and the full algorithm/model/training documentation. |
| **`DeepFilterNet\`** | The upstream [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) engine repo (Rust + Python), plus the pretrained model and a CLI helper (`denoise.bat`). |

```
VoiceDenoiser\
├── AudioDenoiser\            ← the app (run.bat)
│   ├── audio_denoiser_app.py     UI: Live + Files tabs, audio I/O, spectrograms
│   ├── denoiser_pipeline.py      streaming DeepFilterNet3 engine
│   ├── run.bat                   launcher
│   ├── output\                   denoised files land here
│   └── README.md                 docs: how it works, model, training, parameters
├── DeepFilterNet\            ← engine repo (upstream + small local patch)
│   ├── denoise.bat               one-shot CLI: denoise a wav from this folder
│   ├── models\DeepFilterNet3\    pretrained model used by the app
│   └── DeepFilterNet\df\         model code, training, offline enhance.py
└── .venv\                    ← Python 3.12 environment everything runs in
```

## Setup (one time, from scratch)

Tested on Windows 11. Python 3.13/3.14 do **not** work with this project
(pinned `pyo3`/`numpy<2`); use 3.12. Torch must be **≤ 2.8** (2.9+ removed
APIs the engine imports). The Rust toolchain uses the **GNU** target so the
multi-GB Visual Studio Build Tools aren't needed.

```powershell
# 1. Toolchains
winget install Python.Python.3.12
winget install Rustlang.Rustup
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu

# 2. Virtual environment + dependencies (from this folder)
py -3.12 -m venv .venv
.venv\Scripts\pip install torch==2.8.* torchaudio==2.8.* --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install "numpy<2" loguru appdirs requests packaging maturin soundfile scipy pillow matplotlib

# 3. Build the Rust engine extension (libdf) into the venv
cd DeepFilterNet
..\.venv\Scripts\maturin develop --release -m pyDF/Cargo.toml

# 4. Extract the pretrained model
tar -xf models\DeepFilterNet3.zip -C models
```

If `models\DeepFilterNet3\` (with `config.ini` + `checkpoints\`) already exists,
step 4 is done.

## Run

```powershell
# The app (Live mic denoising + Files batch denoising)
AudioDenoiser\run.bat

# One-shot CLI file denoising (output lands in DeepFilterNet\out\)
cd DeepFilterNet
denoise.bat path\to\noisy.wav
```

First launch loads the model (~2–4 s). In the app: pick your mic + output,
flip **ON**. Recommended starting point: Max suppression 24 dB, Block 100 ms,
everything else default — then tune live, every slider applies instantly.
Use headphones while monitoring to avoid feedback.

## Troubleshooting

- **No sound / "No signal from the microphone"**: the app logs everything to
  `AudioDenoiser\audiodenoiser.log`. With Audio Relay, audio only arrives while
  your phone is actively streaming to the Virtual Mic device.
- **ModuleNotFoundError: df**: the `PYTHONPATH` must point at
  `DeepFilterNet\DeepFilterNet` (the inner folder — the repo root is itself
  named DeepFilterNet). `run.bat` and `denoise.bat` handle this for you.
- **torch install errors on Python ≠ 3.12**: recreate the venv with 3.12.
- **Rebuilding libdf**: `maturin develop --release -m pyDF/Cargo.toml` from
  `DeepFilterNet\` (build cache in `DeepFilterNet\target\` makes this fast).
