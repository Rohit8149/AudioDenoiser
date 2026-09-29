import sys
import os

# Add directory to sys.path
sys.path.append(os.path.abspath('AudioDenoiser'))

import tkinter as tk
import customtkinter as ctk
import audio_denoiser_app
from unittest.mock import MagicMock

# Create a mock app
app = audio_denoiser_app.AudioDenoiserApp()
app.mode = "denoise"
app.dn = MagicMock()
app.dn._sv_score = 0.55
app.isolate_var.set(True)
app.live_cocktail_var.set(False)
app.threshold_var.set(0.80)

print("Initial _sv_score:", app.dn._sv_score)
print("Initial threshold:", app.threshold_var.get())

try:
    app._update_biometrics()
    print("UI Update Success!")
    print("Match lbl text:", app.match_val_lbl.cget("text"))
    print("Progress color:", app.match_bar.cget("progress_color"))
except Exception as e:
    print("UI Update Failed:", e)

