# Advanced Machine Learning (AML) Presentation Notes
**Project:** Real-Time AI Audio Denoiser & Biometric Speaker Isolation

---

## 1. What AML Concepts We Used (In Simple Terms)
Even though we used foundational libraries, our application relies entirely on these core Advanced Machine Learning concepts:

* **Deep Learning (Neural Networks):** A system inspired by the human brain that learns patterns. We used this to recognize the complex mathematical patterns of human speech versus the patterns of background noise.
* **Recurrent Neural Networks (RNNs & GRUs):** A special type of neural network that has "memory." Audio is time-series data. To remove noise, the AI needs to remember what the noise sounded like 1 second ago. The Gated Recurrent Units (GRUs) provide this memory.
* **Feature Extraction (Spectrograms):** Changing raw data into a format the AI can easily understand. Instead of feeding the AI raw audio waves, we convert the audio into a "Spectrogram" (a visual heat-map of frequencies) using Short-Time Fourier Transforms (STFT), because AI processes these visual frequency patterns much better.
* **Representation Learning (Vector Embeddings):** Converting something complex into a simple list of numbers. Our app takes a 5-second sample of a person's voice and mathematically compresses it into a 192-number "fingerprint" (a d-vector).
* **Distance Metrics (Cosine Similarity):** A math formula used to check how similar two vectors are. We use Cosine Similarity to compare the live microphone voice against the saved Voice Print. If the math says they point in the same direction, it confirms the speaker's identity.
* **Thresholding & Gating:** Setting a mathematical cutoff point to make a binary decision. We set a threshold of 0.28. If the AI is more than 28% confident it is the target voice, it keeps the audio. If it is less, it applies a heavy attenuation gate.

---

## 2. What We Used (But Did NOT Make)
* **The Pre-Trained Weights (The AI's Brain):** We did not collect thousands of hours of audio and train the neural networks from scratch on a supercomputer. We utilized **Pre-Trained Foundational Models** (DeepFilterNet and SpeechBrain).
* **The Backpropagation Math:** We did not write the complex calculus that teaches the AI how to fix its mistakes during training. PyTorch handles the gradient descent and backpropagation.
* **Generative AI:** We used *Discriminative/Masking* AI, not generative AI. We are calculating noise masks to multiply against real audio, not generating fake audio (like Deepfakes).

---

## 3. What WE Actually Engineered (Our Hard Work)
*How to explain this to the professor: "We didn't train the brain, we built the complex nervous system required to make that brain work in real-time on a standard CPU."*

* **Real-Time Data Pipelining (Edge Computing):** We wrote the Python pipeline that chops live microphone audio into tiny 40-millisecond blocks and feeds them to the AI fast enough that there is zero lag.
* **The Overlap-Add Crossfade:** When you chop audio into blocks, the AI creates ugly "tearing" sounds at the edges. We mathematically engineered a Hanning-window crossfader to stitch the blocks back together perfectly smoothly.
* **Asynchronous Multi-Threading:** We built a background thread that constantly calculates the Cosine Similarity of the speaker *while* the main thread simultaneously cleans the noise. This prevents the real-time audio loop from freezing.
* **Fast-Attack / Slow-Release Smooth Gating:** We wrote an algorithm that smoothly fades the volume down when a stranger talks, instead of harshly chopping the audio in half.

---

## 4. The Audio Profiles & Parameters

We built these sliders to give mathematical control over the AI's output:

* **Noise Attenuation (dB):** How aggressively the AI clamps down on non-speech sounds (100dB = absolute silence, 60dB = leaves natural room tone).
* **Wet/Dry Mix (%):** Blends the raw noisy microphone (0%) with the denoised AI output (100%). We use 95% to bleed a tiny bit of true high-frequency breath sounds back into the audio to prevent it from sounding robotic.
* **Output Gain (dB):** A volume booster (+12dB) applied at the end to make up for the acoustic energy lost when the noise was deleted.
* **Low-cut Filter (HPF - Hz):** Deletes low-end rumble (truck engines, wind) before the AI processes it, saving compute power.
* **Mask Floor (dB):** Sets the maximum penalty the AI can apply to a frequency, preventing it from over-suppressing and destroying the voice.

### The Presets:
1. **Home (Natural Mix):** Gentle settings (40dB mask floor, 95% mix). Perfect for a quiet room where you just want to remove PC fan noise. Keeps the voice sounding warm and broadcast-quality.
2. **Classroom (Babble Control):** Designed for unpredictable human noise. Adds an aggressive **Post-Filter** to catch sudden shrill spikes (like chairs scraping).
3. **Traffic (Max Suppression):** For extreme environments. Raises the HPF to 160Hz to aggressively chop out heavy bass rumble from engines and wind.

---

## 5. Future Scope (Where to take the project next)

* **Custom Fine-Tuning (Transfer Learning):** Collect a custom dataset of audio specifically from the university's lab environments and fine-tune the DeepFilterNet weights using gradient descent so it becomes highly specialized to our exact campus acoustics.
* **Overlapping Speech Separation (PIT Loss):** Build a lightweight separation model (like Conv-TasNet) in PyTorch and train it from scratch using **Permutation Invariant Training (PIT Loss)** to solve the "Cocktail Party Problem" where two people speak at the exact same time.
* **Model Quantization:** Take the heavy 32-bit floating-point (FP32) models and mathematically compress them to 8-bit integers (INT8) to run them even faster on lower-end CPUs without losing accuracy.
