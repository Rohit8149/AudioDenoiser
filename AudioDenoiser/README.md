# AudioDenoiser

Real-time microphone noise suppression for Windows, plus offline batch denoising of
audio files, built on the open-source **DeepFilterNet3** speech-enhancement model
(MIT/Apache-2.0). Two tabs:

- **Live** — your microphone goes in, denoised audio comes out of your speakers in
  real time, with live spectrograms and a panel of live measurements.
- **Files** — drop in noisy recordings (wav, mp3, flac, ogg, m4a, aac, wma) and get
  denoised 48 kHz WAV files back, processed with the full-quality offline path.

## Quick start

Double-click `run.bat`. Requires the Python 3.12 venv at `.venv` (in the repo root)
(setup instructions are in the main repo README).

- **Live tab:** pick Microphone and Output, then use the two switches:

  | Main (denoiser) | Repeat (playback) | What happens |
  |---|---|---|
  | OFF | OFF | nothing |
  | OFF | ON | **raw passthrough** — mic straight to output, unfiltered |
  | ON | ON | **denoised** audio played to output (normal use) |
  | ON | OFF | denoiser runs silently — stats + spectrograms update, output muted |

  Toggling Repeat takes effect immediately, even while running. The Attenuation
  slider (0–100 dB) controls how aggressively noise is removed — 100 dB gives
  maximum silence between words, lower values keep more natural room tone. The
  customization sliders (dry/wet mix, output gain, high-pass filter) apply
  instantly in denoise mode. Block size trades latency vs. quality (see "Live
  vs. offline quality" below).
- **Files tab:** its own sliders, *Add files…* → *Denoise files*. Results land in
  `output\<name>_denoised.wav`. This path uses the same model but processes whole
  files at once, which is slightly better quality than live mode (no block-edge
  artifacts) and much faster than realtime.

Use headphones when monitoring live — loud speakers next to the mic create a
feedback loop. If you feed the mic via [Audio Relay](https://audio-relay.com)
("Virtual Mic" device), remember audio only reaches the denoiser while your phone
is actively streaming.

## Controls reference

Both tabs have the same slider set (independent values per tab):

| Control | What it does |
|---|---|
| Noise Attenuation [dB] | Global cap on how much noise is removed: 100 dB = maximum suppression; 0 = bypass. Applies live — no restart needed |
| **Max suppression [dB]** | **Per-frequency-bin floor — the speech-preservation control.** No single frequency bin may be attenuated more than this, no matter what the model wants. Default **24 dB**. Slide right (`off`) for maximum aggression. |
| Dry / Wet mix | 100 % = fully denoised; lower blends the original mic back in. Applies live |
| Output gain [dB] | Boost/cut the output level; auto guard against clipping. Applies live |
| High-pass filter [Hz] | Cuts rumble/fan hum below the cutoff (off below 20 Hz). Applies live |
| Aggressive post-filter | (Live tab) extra suppression of very noisy sections; reloads the model — makes suppression *more* aggressive, not less |

All sliders take effect immediately, including while the denoiser is running.

### Tuning for "it cuts off my voice when noise starts"

This over-suppression is inherent to speaker-agnostic denoisers: the model
suppresses anything that doesn't look like speech, and speech buried in traffic
can look non-speech-like, so the model silences it along with the noise. The
**Max suppression** floor fixes exactly this — it is enabled by default at
24 dB. Recommended, in order:

1. **Max suppression ≈ 18–24 dB** — guarantees speech peaks stay audible; noise
   is still reduced by up to that amount, which is usually plenty. Set it to
   `off` only when the background is truly quiet.
2. **Block size 200 ms** (the default) — the model sees more context and
   recognizes speech-in-noise much better than at 40 ms.
3. **Dry/Wet ≈ 80–90 %** — leaves a touch of the room in, masking artifacts and
   making the gate far less obvious.
4. Leave **Aggressive post-filter off** — it intentionally over-attenuates.

### Playback continuity (clock drift)

Mic and speakers run on different clocks (especially Audio Relay's virtual mic,
fed over the network). A jitter buffer with a ~2-block cushion bridges them:
playback is continuously micro-resampled by up to ±0.4 % — inaudible — so the
audio never drifts into gaps or stutters. Total monitoring latency is roughly
block size + ~200 ms of cushion + ~30 ms algorithmic.

### Why not just use a "better model"?

DeepFilterNet3 (2.14M parameters) is the largest model this framework ships —
there is no official bigger DFN. Heavier published denoisers (FullSubNet,
MossFormer2, …) need entirely different runtimes and most cannot run 48 kHz
full-band in real time on one CPU core, which is the whole point here. The
suppression floor achieves the practical goal — speech survives noise — without
changing models. The real "better model" upgrade for this use case is the
personalization roadmap item: once only *your* voice is kept, aggressive
suppression stops being risky.

---

# How it works, start to finish

This section explains exactly what happens to the audio, what the model is, how it
was built and trained, and where the training data comes from.

## 1. The signal chain (per 10 ms frame)

Audio is processed in mono at 48 kHz. The live pipeline repeats this sequence for
every block (default 100 ms = 10 frames); the offline path runs the identical model
over the whole file in one pass.

```
mic / file (any rate) ──resample──> 48 kHz mono float32
   │
   1. STREAMING STFT        960-sample FFT (20 ms window), 480-sample hop (10 ms),
   │                        raised-cosine window, implemented as a continuous
   │                        overlap-add loop → 481 complex frequency bins.
   │                        Cost: ~10 ms algorithmic latency (window − hop).
   2. FEATURE EXTRACTION
   │    a. ERB features: the 481 bins are collapsed into 32 perceptual
   │       "equivalent rectangular bandwidth" bands, converted to dB, then
   │       mean-normalized with an exponential moving average (α from a 1 s
   │       time constant). This compresses 481 numbers into 32 that mimic
   │       how the inner ear groups frequencies.
   │    b. Spectral features: the lowest 96 bins are unit-normalized the same
   │       way (running magnitude average), because low frequencies carry the
   │       tonal detail the deep filter needs.
   3. DEEPFILTERNET3 MODEL (2,135,484 parameters, PyTorch, CPU)
   │    outputs two things per frame (see architecture below):
   │      • an ERB gain mask  m ∈ (0,1)^32   — the "noise gate"
   │      • Deep Filter FIR coefficients — 3 complex taps for each of the
   │        96 lowest bins — the "repair filter"
   4. SPECTRAL PROCESSING
   │      • the ERB mask is expanded back to the 481 bins
   │      • Deep Filtering combines the current frame with the 2 previous
   │        frames' spectra through the predicted FIR coefficients, which
   │        removes artifacts a plain gain mask leaves behind (musical noise,
   │        smearing, partial suppression of interfering speech)
   5. OUTPUT CHAIN
   │      • attenuation limiter:  out_spec = spec·lim + enhanced·(1−lim),
   │        lim = 10^(−atten_dB/20)  → the Attenuation slider, 0 = bypass
   │      • suppression floor: no bin may fall more than "Max suppression"
   │        dB below the input magnitude → speech survives heavy noise
   │      • dry/wet mix (in the spectral domain, so both paths are aligned)
   │      • inverse STFT back to time domain (streaming, overlap-add)
   │      • high-pass biquad filter (your slider), output gain, soft anti-clip
   │
   speakers / output WAV
```

Everything the model needs to stay continuous across blocks — the STFT overlap
buffers, both normalization states, and the two GRU hidden states — is persisted
between calls in `denoiser_pipeline.py`. (Stock `df` Python code discards the GRU
state on every call; this app wraps the GRU modules so it carries over, which is
what the original C++/Rust real-time engine does.)

## 2. The model: DeepFilterNet3

DeepFilterNet3 comes from the paper *DeepFilterNet: Perceptually Motivated
Real-Time Speech Enhancement* (Schröter, Rosenkranz, Escalante-B., Maier —
INTERSPEECH 2023, [arXiv:2305.08227](https://arxiv.org/abs/2305.08227)), by the
Machine Learning and Data Analytics (MaD) lab at FAU Erlangen-Nürnberg. It is the
third generation of the DeepFilterNet family ([DF1, ICASSP 2022](https://arxiv.org/abs/2110.05588),
[DF2, IWAENC 2022](https://arxiv.org/abs/2205.05474)).

Architecture (all code in `DeepFilterNet/df/deepfilternet3.py`):

1. **ERB encoder** — three stride-2 grouped convolutions over the 32 ERB features
   with 2-frame lookahead, producing an embedding that summarizes "what kind of
   sound is this frame".
2. **Spectral encoder** — two convolutions over the 96 normalized bins, embedded
   into the same space and added to the ERB embedding.
3. **Squeezed GRU** (1 layer, 256 hidden units, grouped linear in/out) — the
   temporal memory. It models noise dynamics: fans stay constant, keyboards are
   transient, speech has structure.
4. **Two decoders:**
   - the **ERB decoder** (transposed-conv U-Net style with skip connections from
     the encoder) predicts the per-band gain mask m, plus a local-SNR estimate;
   - the **Deep Filter decoder** (a second small GRU + linear heads) predicts the
     complex FIR coefficients for the multi-frame filter.
5. **Deep Filtering** applies those coefficients to the noisy spectrum — a linear,
   distortion-free operation, unlike a gain mask which can only scale bins.
6. **Post-filter** (optional, the "Aggressive" checkbox) — a spectral-comparison
   derived over-subtraction of the noise estimate for extra suppression of very
   noisy segments, at the cost of slight over-attenuation.

DF3's key perceptual tricks vs. DF2: the loss weights low frequencies much more
heavily (where human hearing is most sensitive and most artifacts are audible),
the deep filter operates on more bins, and the post-filter sharpens the noise
floor. The result is the model in this repo: **2.14M parameters, ~180 MFLOPs/s**,
which is why it runs in real time on one CPU core — here at ~13 ms per 100 ms of
audio (0.13× realtime) on the machine this was developed on.

The model is **speaker-agnostic**: it was trained to recognize *speech* as a
signal class, not any particular voice. That's why it keeps *other people's*
speech too — and why personalized ("only my voice") denoising is the next planned
feature.

## 3. How the model was trained

Training entry point: `DeepFilterNet/df/train.py` with data prepared by
`df/scripts/prepare_data.py`. The recipe, in sequence:

1. **Collect corpora** (see datasets below) and convert everything to 48 kHz
   mono PCM, then pack into HDF5 databases — one each for speech, noise and
   room impulse responses (RIRs), split into train/validation/test.
2. **Dynamic mixing** (not pre-mixed!): at training time each sample is created
   on the fly — a speech utterance is convolved with a random RIR (simulating a
   room), a noise segment is added at a random signal-to-noise ratio, with local
   SNR values spanning **−15 dB … +35 dB** (`LSNR_MIN`/`LSNR_MAX` in the config).
   This means the model effectively sees an infinite number of noisy mixtures
   and never overfits to specific noise clips.
3. **Augmentation**: random gains, codec/bandwidth augmentation (the model is
   trained on full-band 48 kHz audio but must also handle band-limited input),
   per-sample alignment of speech/noise pairs.
4. **Loss**: a weighted combination of
   - a **multi-resolution STFT loss** (L1 on magnitudes at several FFT sizes) and
     a frequency-weighted spectral loss, both applied **with DF3's perceptual
     weighting** that emphasizes low frequencies ~20× more than high ones,
   - a **mask loss** (L1 between predicted and ideal ERB gain), and
   - a **deep-filter SNR loss** — the signal is reconstructed through the
     predicted deep filter and compared to the clean target directly, so the
     filter coefficients are trained end-to-end.
5. **Optimization**: Adam with a one-cycle LR schedule (see `df/lr.py`), batch
   size 32, ~120 epochs to the released "best" checkpoint (this app loads epoch
   120 from `DeepFilterNet/models/DeepFilterNet3/`).

## 4. Training datasets (and where to get them)

The released models follow the DeepFilterNet paper recipe, built on the
**Microsoft DNS-Challenge** (Deep Noise Suppression Challenge) corpora — the
same data used for DNS-4/DNS-5. The repo's `scripts/download_process_dns4.sh`
automates the download+conversion: it pulls Microsoft's
`download-dns-challenge-4.sh` blob list from the
[DNS-Challenge GitHub](https://github.com/microsoft/DNS-Challenge) and converts
each tarball into HDF5 (`prepare_data.py` with types `speech`, `noise`, `rir`).

| Dataset type | Contents | Origin / license |
|---|---|---|
| **Speech** (~750 h) | Read English speech: LibriVox (public domain), VCTK, custom recordings, out-of-domain validation sets | DNS-Challenge blobs `read_speech`, `vctk_wav48_ssd` |
| **Noise** (~180 h) | Freesound clips (CC), DEMAND environmental recordings, audiolab noises, transformed noise | DNS-Challenge blob `noise` |
| **RIRs** (~2 k rooms) | Room impulse responses: real (OPENAIR and similar) + synthetic (image method) | DNS-Challenge blob `impulse_responses` |

All are freely downloadable from the DNS-Challenge repository (Azure blob links
in its `download-dns-challenge-4.sh`). To retrain: download →
`scripts/download_process_dns4.sh` → `df/scripts/prepare_data.py` → write a
`dataset.cfg` (example in `assets/dataset.cfg`) → `python df/train.py
dataset.cfg data_dir base_dir`. Training is GPU-bound (a modern 12 GB GPU
trains DF3 in a few days) and realistically needs Linux.

## 5. Live vs. offline quality

The Python model applies its convolution lookahead (2 frames) inside each call,
so frames at the edge of a live block lack future context (the Rust real-time
engine buffers this properly; the Python path can't). Measured output
correlation against the offline enhancer on the test file:

| Live block size | Latency | Correlation with offline |
|---|---|---|
| 40 ms | ~60 ms | 0.83 |
| 100 ms (default) | ~120 ms | 0.93 |
| 200 ms | ~220 ms | 0.97 |
| 400 ms | ~420 ms | 0.995 |

**Files tab always uses the offline path** — the whole file is processed in a
single pass (verified bit-for-bit equivalent in correlation to the official
`enhance()` reference, r = 1.000 at lag 0), so it gets full model context and
the best possible quality.

## 6. Measurements from this machine

- Live inference: ~7–13 ms per block → **5–14× faster than realtime** on CPU
  (6 physical cores; torch thread count = physical cores is optimal — 12
  hyperthreads is actually slower)
- File denoising: **~21× realtime** (a 60 s file takes ~3 s, a 5 min recording
  ~15 s), processed in 4 s chunks — full-context quality, 0.997 correlation
- Latency (Live monitoring): block size + ~30 ms algorithmic + ~150 ms jitter
  cushion ≈ **280 ms total** at the default 100 ms block (40 ms block ≈ 220 ms)
- Model: DeepFilterNet3, epoch 120, 2,135,484 parameters, 32 ERB bands,
  96 deep-filter bins, FFT 960 / hop 480 @ 48 kHz, ~20 ms algorithmic latency

## 7. All parameters explained

### GUI parameters (both tabs unless noted)

| Parameter | Range / default | Exactly what it does |
|---|---|---|
| **Main ON/OFF** | — | Starts/stops the processing engine. OFF + Repeat ON = raw passthrough (no model, no filters, mic straight to output) |
| **Repeat ON/OFF** | ON | Output playback switch. ON + denoiser OFF = passthrough; denoiser ON + repeat OFF = silent processing (spectrograms/stats still run) |
| **Noise Attenuation [dB]** | 0–100, default 100 | Global output limiter applied in the STFT domain: `out_spec = spec·lim + enhanced·(1−lim)` with `lim = 10^(−dB/20)`. 100 dB ≈ lim 0 → fully enhanced signal; 0 dB → lim 1 → raw signal. This is a *linear blend toward the noisy input*, so it also restores everything the model removed. Applies live. |
| **Max suppression [dB]** | 0–100, default 24 | Per-bin magnitude floor applied after the limiter: any bin whose output magnitude fell below `input_magnitude · 10^(−dB/20)` is raised to that floor (input phase from the enhanced signal is kept). Bounds how much any single frequency can be gated — the speech-preservation control. 100 = no floor. Applies live. |
| **Dry / Wet mix [%]** | 0–100, default 100 | Second blend stage: `out = enhanced·mix + spec·(1−mix)` in the spectral domain (perfectly aligned, no comb filtering). 80–90 % leaves natural room tone. Applies live. |
| **Output gain [dB]** | −12…+12, default 0 | Time-domain gain after synthesis, then anti-clip guard (scales down if peak > 1.0). Applies live. |
| **High-pass filter [Hz]** | 0–400 step 20, default 0 | 2nd-order Butterworth high-pass (scipy `sosfilt`, zero-phase-free streaming with persistent filter state). Kills rumble, fan hum, wind below the cutoff. Applies live. |
| **Block size [ms]** | 40/100/200/400, default 100 | How much audio the live loop processes per pass. Bigger = model sees more context (better speech-in-noise recognition, higher fidelity: corr 0.83/0.93/0.97/0.995 vs offline at 40/100/200/400 ms) but more latency. Applies at next ON. |
| **Aggressive post-filter** | off, Live tab | Enables DFN3's `mask_pf` spectral post-filter at model load (model reload ~2 s): over-subtracts the noise estimate in very noisy segments. More suppression, less natural — leave off for speech preservation. |
| **Microphone / Output** | device list | Any input/output endpoint, including Audio Relay's "Virtual Mic". Changes apply at next ON. |

The Live and Files tabs store **independent** slider values; file processing
applies the Files-tab values, live monitoring the Live-tab values.

### Internal (model) parameters — from `models/DeepFilterNet3/config.ini`

| Parameter | Value | Role |
|---|---|---|
| `fft_size` / `hop_size` | 960 / 480 @ 48 kHz | 20 ms STFT window, 10 ms frame hop, 481 bins, ~10 ms algorithmic latency |
| `nb_erb` | 32 | ERB bands the mask operates on (perceptual compression of 481 bins) |
| `nb_df` | 96 | Lowest bins the Deep Filter re-synthesizes |
| `df_order` | 3 | Complex FIR taps per bin (2×3 real coefficients predicted per bin/frame) |
| `df_lookahead` / `conv_lookahead` | 2 / 2 | Future frames the encoders may use (why bigger blocks help) |
| `emb_hidden_dim` / `df_hidden_dim` | 256 / 256 | GRU widths of encoder and deep-filter decoder |
| `lsnr_min` / `lsnr_max` | −15 / +35 dB | Local-SNR headroom of the training mixtures |
| normalization α | ≈ 0.99 | Exponential moving average of the feature normalizers (1 s time constant) |
| weights | epoch 120, 2,135,484 params | `models/DeepFilterNet3/checkpoints/model.ckpt` |
| jitter buffer | target 1.5 blocks, ±0.4 % | Playback bridge between mic and speaker clocks (see "Playback continuity") |

## 8. Files in this folder

| File | Purpose |
|---|---|
| `audio_denoiser_app.py` | Tkinter UI: Live + Files tabs, sounddevice audio I/O, spectrogram rendering, live stats |
| `denoiser_pipeline.py` | `StreamingDenoiser`: streaming STFT, persistent feature-normalization and GRU states, attenuation limiter, dry/wet, high-pass, gain, `denoise_file()` |
| `run.bat` | Launcher |
| `audiodenoiser.log` | Runtime log (levels, dropouts, errors) — check here when something misbehaves |
| `output\` | Denoised files land here |

## 9. Roadmap

1. **Personalized denoising** — enroll your voice, then gate out *other* speakers
   (personal VAD / speaker-embedding gate on top of DFN's output).
2. **Adaptive noise tuning** — detect the current noise type and auto-adjust
   attenuation and post-filter strength per band (runtime parameters already
   exist for this).
3. **Windows virtual microphone** — WASAPI capture → DFN → virtual device, so
   Teams/Meet/Discord can select "AudioDenoiser" directly.
