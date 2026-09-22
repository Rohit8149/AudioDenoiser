# 🎙️ AudioDenoiser: Real-Time AI Streaming Noise Suppression

**AudioDenoiser** is a Python-based, real-time AI audio processing application designed to hijack microphone input, apply deep-learning-based noise suppression, and route the clean audio to a Virtual Audio Cable for system-wide use (e.g., in Zoom, Teams, or OBS).

This project adapts the highly optimized **DeepFilterNet3** architecture for native Python streaming, featuring a custom **Overlap-Add Crossfader** to achieve mathematically flawless, tear-free audio processing in real-time.

---

## 🚀 Key Features

*   **Real-Time AI Denoising:** Utilizes DeepFilterNet3 (PyTorch) to suppress background noise, keyboard clicks, and environmental sounds.
*   **Zero-Tearing Overlap-Add Pipeline:** A custom mathematical crossfader that completely eliminates boundary discontinuities (the "kr kr" tearing effect) inherent in PyTorch convolutional streaming.
*   **Custom Siren/Horn Killer:** A spectral median-filter post-processing step designed to catch and duck high-frequency, sudden transients (like traffic horns and sirens) that evade standard AI models.
*   **System-Wide Audio Routing:** Seamlessly intercepts microphone audio and outputs it to a Virtual Audio Cable using PyAudio.
*   **Interactive UI:** Built with `customtkinter`, featuring real-time waveform visualization, output gain control, and bypass toggles.
*   **[UPCOMING] Target Speaker Extraction (Voice Printing):** A planned future feature that will allow the system to record a sample of the user's voice and *cancel out all other human voices* (solving the "cocktail party problem" and background TV/News broadcasts).

---

## 🧠 For the Research Paper & PPT: Core Concepts & Knowledge Base

If you are building a presentation or research paper on this project, here are the critical technical concepts you need to understand and mention:

### 1. The Core AI Architecture (DeepFilterNet3)
*   **STFT (Short-Time Fourier Transform):** The audio is converted from waveforms into spectrograms (frequencies over time) before the AI processes it.
*   **ERB (Equivalent Rectangular Bandwidth):** The AI maps frequencies to human hearing scales to process sound the way human ears perceive it.
*   **GRU (Gated Recurrent Units) + Conv2d:** The neural network uses GRUs (for temporal memory/context) and 2D Convolutions (to find patterns in the spectrogram). 

### 2. The Streaming Challenge & The "Holy Grail" Fix (Highlight This!)
*   **The Problem ("Convolutional Zero-Padding Artifacts"):** Standard PyTorch `Conv2d` layers are designed for offline files. When fed real-time 40ms audio blocks, PyTorch constantly zero-pads the boundaries. This causes a massive 25Hz discontinuity (a "kr kr" chainsaw tearing sound) because the AI loses context at the edge of every chunk.
*   **The Solution (Overlap-Add Crossfading):** To achieve flawless real-time streaming in Python, we implemented a **120ms Overlap-Add Crossfader**. 
    *   The system buffers 3 blocks (120ms) of audio.
    *   It passes the entire 120ms to the AI so the neural network has full context.
    *   It extracts the 40ms target block from the center.
    *   It applies a **Hanning Window** crossfade to seamlessly blend the overlapping boundaries together. 
    *   *Result:* 0% tearing, flawless neural network memory, and mathematically continuous audio.

### 3. Audio Hijacking & Routing
*   **PyAudio / PortAudio:** The library used to interface directly with Windows audio drivers (WASAPI/MME).
*   **Virtual Audio Cable (VAC):** Software that acts as a digital patch cable. AudioDenoiser captures the physical microphone, cleans it, and plays it into the VAC input. Communication apps (Zoom, Discord) then read from the VAC output.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    Mic[Physical Microphone] -->|PyAudio Stream| Q_In[Input Queue]
    Q_In --> Buffer[120ms Rolling Buffer]
    
    subgraph AI Pipeline (denoiser_pipeline.py)
        Buffer --> STFT[Complex STFT]
        STFT --> AI[DeepFilterNet3 PyTorch Model]
        AI --> Mask[Spectral Masking]
        Mask --> Siren[Siren & Horn Killer]
        Siren --> ISTFT[Inverse STFT]
        ISTFT --> Crossfade[Hanning Overlap-Add Crossfader]
    end
    
    Crossfade --> Q_Out[Output Queue]
    Q_Out -->|PyAudio Stream| VAC[Virtual Audio Cable Output]
    VAC --> Zoom[Zoom / Teams / OBS]
```

---

## 🔮 Future Work: Target Speaker Extraction
While the current AI excels at removing non-human background noise, our next major architectural update will tackle the **Cocktail Party Problem** (e.g., background news broadcasts or people talking in the same room). 

**How it will work:**
1.  **Voice Enrollment:** The user records a 5-second sample of their clean voice.
2.  **Speaker Embedding (d-vector):** A speaker verification model extracts the unique biometric "print" of the user's voice.
3.  **Conditioned Masking:** The embedding vector is fed into the denoiser network alongside the audio, forcing the AI to treat *any voice that doesn't match the embedding* as noise.

---

## 💻 Setup & Execution

**Prerequisites:**
1. Python 3.12+
2. Virtual Audio Cable installed (e.g., VB-Cable).

**Running the Application:**
Simply execute the batch script which activates the virtual environment and launches the UI:
```cmd
run.bat
```
