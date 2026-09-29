# How AudioDenoiser Works

This document explains how the app clears background noise, using simple examples!

## The Brain: DeepFilterNet3 (The AI Bouncer)
At the core of this app is an Artificial Intelligence model called **DeepFilterNet3**. 
Think of this AI like a very smart bouncer at a nightclub. This bouncer has listened to thousands of hours of audio and has learned exactly what a **human voice** sounds like, and exactly what **background noise** (like keyboard typing, dogs barking, or a fan humming) sounds like.

When you speak into your microphone, the audio goes to the bouncer. The bouncer instantly recognizes your voice and lets it through, but recognizes the dog barking and blocks it. 

---

## The Controls: Fine-Tuning the Bouncer

Here is what the sliders in the app actually do to control the AI:

### 1. Noise Attenuation (How strict the bouncer is)
* **What it does:** This tells the AI how aggressively to block the noise. 
* **Example:** If you set it to **100 dB** (maximum), the AI will try to make the background completely silent. If you set it to **10 dB**, it will only quiet the background noise down a little bit, but still let some of it through.

### 2. Dry / Wet Mix (Blending the audio)
* **What it does:** "Dry" means the raw, noisy audio from your mic. "Wet" means the 100% cleaned, AI-processed audio. This slider lets you mix them together.
* **Example:** Sometimes, if the AI removes *too much* noise, your voice might sound slightly robotic or artificial. By setting the mix to **90%**, you are adding just a tiny sprinkle (10%) of the original room noise back in. This often makes your voice sound much more natural to the listener, while still being very clean.

### 3. Output Gain (The volume knob)
* **What it does:** Simply makes the final output louder or quieter. 
* **Example:** When the AI removes loud background noise, the overall volume of your audio might drop. You can turn the Gain up (e.g., +3 dB) to make your voice louder again.

### 4. High-Pass Filter (The rumble blocker)
* **What it does:** This doesn't use AI. It is a traditional filter that blocks very low, deep sounds (low frequencies) from passing through. 
* **Example:** If you accidentally bump your desk, or if wind blows on your microphone, it creates a loud, deep "thump" or rumble. Setting this filter to **80 Hz** is like putting a net over the microphone that catches those deep thumps, but lets your higher-pitched voice pass right through.

### 5. Max Suppression / Floor (Avoiding the "Vacuum" effect)
* **What it does:** This puts a limit on how quiet the background is allowed to get.
* **Example:** If the background is perfectly, 100% silent when you stop speaking, it can feel very unnatural to the person listening to you (like you suddenly dropped off the phone call). Setting the floor to **24 dB** tells the AI: *"Remove the noise, but leave just a tiny, quiet whisper of room sound so the listener knows the call is still connected."*

### 6. Post-Filter (The extra scrub)
* **What it does:** This is a checkbox you can turn on. It runs the audio through one final cleaning step to catch any tiny bits of noise the main AI missed. It uses a bit more computer power, but results in a cleaner sound.

---

*In short: The AI recognizes your voice and cuts out the rest, and the sliders just let you adjust how natural or aggressive that cut is!*
