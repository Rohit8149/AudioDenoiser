# DeepFilterNet — Voice / Noise Denoiser

DeepFilterNet is a **deep-learning noise suppressor for speech**. You feed it audio that
contains a human voice mixed with background noise (fans, traffic, keyboard, wind, music,
other people talking), and it returns audio with the noise strongly attenuated while the
speech stays clear. It works on **full-band 48 kHz audio**, runs **in real time on a normal
CPU** (a few percent of one core), and is used as a lightweight alternative to heavy
denoisers in calls, recordings, and streaming.

The repo is based on the open-source [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
project (MIT / Apache-2.0 dual licensed), with a small local patch in `libDF/src/capi.rs`
that makes the C API honor the requested channel count.

## How it works (in one minute)

1. **STFT** — the incoming audio is cut into overlapping 10 ms frames (480 samples hop,
   960-sample FFT) and converted to a spectrogram. This introduces ~20 ms of algorithmic
   latency, which is what makes real-time processing possible.
2. **ERB bands** — the 48k spectrogram bins are grouped into perceptually-motivated
   ERB (equivalent rectangular bandwidth) bands, so the network works on ~65 bands instead
   of 481 raw bins. This is the main trick that keeps it cheap.
3. **Two networks**:
   - a small **ERB encoder/GRU decoder** that estimates a noise-gate-like gain per band, and
   - a **Deep Filter**, a linear FIR complex filter (3 taps × 2x48 coefficients) that
     re-synthesizes the signal and also removes distortions the gain stage can't (e.g.
     partial suppression of other speech).
4. **ISTFT** — the filtered spectrum is converted back to time-domain audio, frame by frame,
   in a streaming loop.

The model is speaker-agnostic: it was trained to recognize *speech* as a class, not *your*
voice specifically. That distinction matters for the improvement ideas below.

The project contains three pretrained model generations — **DeepFilterNet**, **DF2**, and
**DF3** — with DF3 being the best. Pretrained weights ship in `models/`.

## Repository layout

| Directory | What it is |
|---|---|
| `libDF/` | Core Rust library: STFT/ISTFT loop, inference backend (Tract), real-time state |
| `pyDF/` | Python binding (`libdf`) around the Rust processing loop, built with maturin |
| `DeepFilterNet/` | Python package (`df`): PyTorch model definitions, `enhance.py`, training, evaluation |
| `pyDF-data/` | PyTorch dataloaders for training (Linux-only features) |
| `models/` | Pretrained models (DeepFilterNet, DF2, DF3; ONNX + tar.gz variants) |
| `ladspa/` | LADSPA plugin for real-time mic denoising via PipeWire (Linux only) |
| `demo/` | Real-time demo apps (GUI + command line) built on `cpal` audio I/O |
| `scripts/` | Utilities: `demo.py` (live mic with Tk UI), `external_usage.py`, perf scripts |

## Running it on Windows 11

Two independent paths exist: a **Python path** (easiest, CPU or GPU) and a **Rust path**
(fastest, no Python). Both work on Windows 11; only the PipeWire/LADSPA real-time mic path
and training are Linux-only.

### Path A: Python (recommended for experiments)

Prerequisites: install [Python 3.10+](https://python.org) (tick "Add to PATH") and run:

```powershell
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install deepfilternet
```

Denoise a file (output lands in `out/`):

```powershell
deepFilter path\to\noisy_audio.wav
# add --pf for extra aggressive attenuation of very noisy sections
# add --output-dir somewhere to choose the output folder
```

From a Python script:

```python
from df import init_df, enhance

model, df_state, suffix, epoch = init_df()   # loads DeepFilterNet3
enhanced = enhance(model, df_state, noisy_audio)   # float32 @ 48 kHz
```

### Path A2: Running from this repository (verified on Windows 11)

The repo layout needs Python 3.10–3.12 (3.13/3.14 are too new for the pinned pyo3/numpy),
torch ≤ 2.8 (2.9+ removed `torchaudio.backend`), and a Rust toolchain to build the `libdf`
extension. One-time setup:

```powershell
winget install Python.Python.3.12
winget install Rustlang.Rustup
rustup toolchain install stable-x86_64-pc-windows-gnu   # avoids the MSVC/VS Build Tools requirement
rustup default stable-x86_64-pc-windows-gnu

# from the repository root (parent of this folder)
python3.12 -m venv .venv
.venv\Scripts\pip install torch==2.8.* torchaudio==2.8.* --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install "numpy<2" loguru appdirs requests packaging maturin soundfile
.venv\Scripts\maturin develop --release -m pyDF/Cargo.toml   # builds and installs libdf

# extract the pretrained model once (creates models\DeepFilterNet3\ with config.ini + checkpoints)
tar -xf models\DeepFilterNet3.zip -C models
```

Then denoise files via the included helper (from this folder):

```powershell
.\denoise.bat path\to\noisy.wav          # output lands in out\
```

or directly:

```powershell
$env:PYTHONPATH = "."
..\.venv\Scripts\python.exe DeepFilterNet\df\enhance.py --model-base-dir models\DeepFilterNet3 -o out noisy.wav
```

Working from this repository instead of PyPI:

```powershell
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install maturin poetry
poetry -C DeepFilterNet install -E train -E eval
maturin develop --release -m pyDF/Cargo.toml
python DeepFilterNet/df/enhance.py -m DeepFilterNet3 path\to\noisy_audio.wav
```

### Path B: Rust binary (no Python)

Grab `deep-filter-x86_64-pc-windows-*` from the
[releases page](https://github.com/Rikorose/DeepFilterNet/releases/), or build it (needs
[rustup](https://rustup.rs)):

```powershell
cargo build --release -p deep-filter
target\release\deep-filter.exe noisy_audio.wav --pf -o out
```

Only 48 kHz .wav input is supported; the output is written to `out/`.

### Real-time microphone denoising

- **The GUI demo (`df-demo`) is Linux-only** per the upstream README. The command-line
  variant `df-demo-c` uses `cpal`, which supports Windows WASAPI — building it on Windows
  is expected to work: `cargo run -p df-demo --bin df-demo-c --release`.
- `scripts/demo.py` gives a live mic-in / speaker-out Tk app (needs the Python path plus
  `pyaudio`).
- The `ladspa/` virtual-microphone setup requires PipeWire (Linux); on Windows the
  equivalent would be a small app that reads the mic with WASAPI, runs `init_df()` in a
  loop, and re-exposes it as a virtual device (this is one of our roadmap items).

### Testing that it works

1. Run the enhancement on any noisy 48 kHz wav (`deepFilter in.wav`) and listen to `out/`.
2. Quality metrics: `DeepFilterNet/df/evaluation_utils.py` computes PESQ/STOI-style
   metrics; `scripts/WAcc*.py` measure word accuracy of Whisper transcription on denoised
   audio — a practical "does it still transcribe me correctly" test.
3. Latency/throughput: `scripts/perf_df_dec.sh` and friends profile the Rust inference loop.

## Roadmap / ideas to improve this

1. **Personalized (speaker-targeted) denoising** — currently the model keeps *any* speech;
   a co-worker talking next to you is preserved. Options, roughly in order of effort:
   - Lightweight post-gate: enroll ~30 s of the user's voice, then on each frame compare
     the (already denoised) frame against a speaker embedding (e.g. ECAPA/GE2E) and
     attenuate frames that look like *other* speech. No retraining of DFN needed.
   - Re-train the enhancement model conditioned on a speaker embedding of the target
     speaker (personal VAD + target-speaker extraction, as done in hearing-aid research).
   - Cheapest first step: a *personal VAD* — only classify "is this frame the user?" and
     let DFN's gain be multiplied by that probability.
2. **More noise types + live auto-tuning** — the model already handles many noise types
   generically; what is missing is (a) including more data with dogs, fans, wind, etc. when
   fine-tuning, and (b) an adaptive runtime that detects the current noise type/spectrum
   and adjusts the attenuation limit and post-filter strength per band automatically
   (DFN already exposes runtime parameters like `atten_lim` and thresholds in
   `RuntimeParams`, so this is an algorithmic addition rather than a retrain).
3. **A Windows virtual microphone app** — a small tray app: WASAPI capture → DFN → virtual
   output device, so Zoom/Discord/Teams can select "DeepFilter Mic" natively.
4. **GPU / ONNX backends** — the ONNX models in `models/` allow deployment outside the
   Rust/Tract path (e.g. DirectML on Windows).

## Citation

The framework is described in three papers: DeepFilterNet ([ICASSP 2022](https://arxiv.org/abs/2110.05588)),
DeepFilterNet2 ([IWAENC 2022](https://arxiv.org/abs/2205.05474)), and DeepFilterNet3
([INTERSPEECH 2023](https://arxiv.org/abs/2305.08227)); multi-frame filtering for hearing
aids in [arXiv:2305.08225](https://arxiv.org/abs/2305.08225).

## License

All code in this repository is dual-licensed under either the MIT License
([LICENSE-MIT](LICENSE-MIT)) or the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)),
at your option.
