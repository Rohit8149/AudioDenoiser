# AML Lab: Distributed AI Audio Denoiser

A real-time, deep-learning audio denoiser and biometric voice isolation tool built for the AML Lab Presentation. This project features a **Triple-AI Pipeline** (DeepFilterNet3 + ECAPA-TDNN) and a **Global Admin Dashboard** (MQTT) allowing judges to remotely control the AI on multiple distributed laptops.

## 🚀 Features
- **Real-Time Denoising:** Suppresses background noise (Traffic, Classroom Babble, etc.) using DeepFilterNet3.
- **Biometric Voice Isolation:** Uses Cosine Similarity embeddings to map and lock onto the speaker's unique vocal tract, filtering out unauthorized voices.
- **Global Remote Control:** Connect multiple PCs across the world. An event-driven Admin Dashboard (HTML/JS) automatically discovers connecting PCs and allows remote control of the AI pipeline.
- **Virtual Audio Cable Integration:** Directly routes the cleaned AI audio into Microsoft Teams or Zoom.

---

## 🛠️ Installation Guide (For PC 2 & PC 3)

### 1. Prerequisites
- **Python 3.10+** installed on your system.
- An **NVIDIA GPU** (Recommended) or a fast modern CPU.

### 2. Download and Setup
Open your terminal (PowerShell) and run these commands:

```powershell
# 1. Clone the repository
git clone https://github.com/Rohit8149/AudioDenoiser.git
cd AudioDenoiser

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\activate

# 4. Install the required AI libraries (This will install PyTorch)
pip install -r requirements.txt
```

### 3. Install the Virtual Audio Cable (Required for Teams/Zoom)
To route the clean audio into your video calls, you need the Virtual Audio Cable:
1. Download it here: [VB-Audio Cable](https://vb-audio.com/Cable/)
2. Extract the ZIP file.
3. Right-click **`VBCABLE_Setup_x64.exe`** and select **Run as Administrator**.
4. Click "Install Driver" and restart your PC.

---

## 🎮 How to Run

### Starting the AI App
Simply run the batch file provided:
```powershell
.\AudioDenoiser\run.bat
```
1. Select your physical microphone as the **Microphone**.
2. Select **CABLE Input** as the **Output**.
3. Type a unique **Device ID** (e.g., `pc2`) and click **Connect to Cloud**.

### Starting the Admin Dashboard
To open the remote control panel:
1. Open the `AudioDenoiser` folder in Windows File Explorer.
2. Double-click the `admin_dashboard.html` file.
3. It will open in Google Chrome and automatically discover all connected laptops!

---

## 📞 Connecting to Microsoft Teams / Zoom
1. Open Teams or Zoom.
2. Go to **Audio Settings**.
3. Change your **Microphone** to **`CABLE Output (VB-Audio Virtual Cable)`**.
4. Now, your colleagues will only hear perfectly cleaned, AI-processed audio!
