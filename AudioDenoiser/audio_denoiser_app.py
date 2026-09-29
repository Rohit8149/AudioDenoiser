"""AudioDenoiser — real-time microphone noise suppression for Windows.

Two tabs:
  * Live      — mic capture -> DeepFilterNet3 -> speaker output, with
                attenuation, sound customization, spectrograms and live stats.
  * Files     — batch-denoise audio files offline (full-quality path).

Run:  run.bat
Use headphones to avoid feedback from speakers back into the mic.
"""

# === Standard library ===
import logging
import os
import queue
import sys
import json
import threading
import time
import traceback
import tkinter as tk
import math
from logging.handlers import RotatingFileHandler

# === Third party ===
import customtkinter as ctk
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageTk
import sounddevice as sd

# === Local ===
from denoiser_pipeline import REPO_DIR, StreamingDenoiser

# ─── Constants ───────────────────────────────────────────────────────────────
# The DeepFilterNet python package bundles the model weights.
MODEL_DIR = os.path.abspath(os.path.join(REPO_DIR, "..", "DeepFilterNet", "models", "DeepFilterNet3"))
HIST_FRAMES = 400      # spectrogram history length in frames (10 ms each)
IMG_W, IMG_H = 820, 130
UI_FPS_MS = 100

JB_MAX_SAMPLES = 480_000       # trim consumed head every ~10 s at 48 kHz
JB_CORRECTION = 0.004          # jitter buffer drift correction factor (±0.4%)
JB_RATIO_MIN = 1.0 - JB_CORRECTION
JB_RATIO_MAX = 1.0 + JB_CORRECTION

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audiodenoiser.log")
_logger = logging.getLogger("AudioDenoiser")
_logger.setLevel(logging.DEBUG)
_log_fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
_fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
_fh.setFormatter(_log_fmt)
_logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(_log_fmt)
_logger.addHandler(_ch)


def log(msg: str) -> None:
    _logger.info(msg)


# ─── Theme ───────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

COLOR_ON = "#22c55e"
COLOR_ON_HOVER = "#16a34a"
COLOR_OFF = "#ef4444"
COLOR_OFF_HOVER = "#dc2626"
COLOR_ACCENT = "#0ea5e9"
COLOR_DIM = "gray55"
COLOR_CARD_INNER = ("gray85", "gray17")

BG = '#0D0D12'
PANEL = '#1A1A24'
PANEL_BORDER = '#2A2A35'
CYAN = '#00E5FF'
GREEN = '#00FF88'
CRIMSON = '#FF3366'
TEXT_PRIMARY = '#E8E8F0'
TEXT_MUTED = '#6B6B80'
TEXT_DIM = '#44445A'


MAGMA = (colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

# defaults for each tab's parameter set.  floor=24 dB keeps speech audible
# through noise (see README "Tuning"); 100 would mean unlimited suppression.
DEFAULT_PARAMS = {"atten": 100, "mix": 100, "gain": 12, "hpf": 0, "floor": 24}


# ─── JitterBuffer ────────────────────────────────────────────────────────────
class JitterBuffer:
    """Bridges two audio devices running on different clocks (e.g. Audio Relay's
    virtual mic vs. a local sound card).  The playback callback pulls a fixed
    number of frames per tick while production drifts against it; without
    compensation you get periodic silence gaps (underrun) or dropped blocks.

    read() continuously micro-resamples (linear interpolation, ±0.4 % max) so
    the consumption rate tracks the fill level around `target` samples.  Real
    clock drift is tens of ppm, so the correction is inaudible; the buffer
    depth absorbs network jitter.
    """

    def __init__(self, target_samples: int):
        self.buf = np.zeros(0, dtype=np.float32)
        self.pos = 0.0  # read position (float, fractional-sample accurate)
        self.target = float(target_samples)

    def append(self, x: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        if self.pos > JB_MAX_SAMPLES:  # trim consumed head every ~10 s
            cut = int(self.pos)
            self.buf = self.buf[cut:]
            self.pos -= cut
        # Safety cap: if buffer grows beyond 2x max, force trim to prevent OOM
        if len(self.buf) > JB_MAX_SAMPLES * 2:
            keep = int(max(0, len(self.buf) - self.target * 2))
            self.buf = self.buf[keep:]
            self.pos = max(0.0, self.pos - keep)

    def read(self, frames: int) -> np.ndarray:
        fill = len(self.buf) - self.pos
        # consumption ratio: pull faster when buffer is full, slower when
        # starved.  Correction is tiny (±0.4 %) — clock drift is tens of ppm,
        # so anything larger is audible pitch wobble.
        ratio = float(np.clip(1.0 + (fill - self.target) / self.target * JB_CORRECTION,
                              JB_RATIO_MIN, JB_RATIO_MAX))
        idx = self.pos + np.arange(frames) * ratio
        if idx[-1] > len(self.buf) - 1:
            # underrun: emit what exists, pad silence, and re-anchor at the end
            out = np.zeros(frames, dtype=np.float32)
            n_good = max(0, int(len(self.buf) - 1 - self.pos))
            if n_good > 0:
                good_idx = self.pos + np.arange(n_good) * ratio
                out[:n_good] = np.interp(good_idx, np.arange(len(self.buf)), self.buf)
            self.pos = max(0.0, float(len(self.buf) - 1))
            return out
        out = np.interp(idx, np.arange(len(self.buf)), self.buf).astype(np.float32)
        self.pos = float(idx[-1]) + ratio
        return out

    def clear(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self.pos = 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═════════════════════════════════════════════════════════════════════════════

class AudioDenoiserApp:

    def __init__(self, root: ctk.CTk):
        self.root = root
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.saved_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    self.saved_config = json.load(f)
            except Exception:
                pass
        root.title("NeuroAcoustic Engine")
        root.geometry("1280x720")
        root.resizable(True, True)
        root.minsize(1280, 720)
        root.configure(fg_color=BG)
        

        # ── state (non-UI) ───────────────────────────────────────────────
        self.pf = False                                   # post-filter on/off
        self.params = {"live": dict(DEFAULT_PARAMS),      # per-tab param values
                       "file": dict(DEFAULT_PARAMS)}
        self.running = False
        self.repeat = False                               # start with repeat OFF
        self.mode: str | None = None                      # None | 'denoise' | 'bypass'
        self.in_stream = None
        self.out_stream = None
        self.worker = None
        self.in_q: queue.Queue = queue.Queue(maxsize=25)
        self.jb: JitterBuffer | None = None
        self.jb_lock = threading.Lock()
        self._dn_lock = threading.Lock()
        self.overruns = 0
        self.block_ms = 100
        self.start_time = None
        # Recording state: captures raw mic + AI output simultaneously
        self._recording = False
        self._rec_raw = []
        self._rec_ai = []
        self.remote_controller = None
        self.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        self._file_busy = False

        # ── build UI, then load model ────────────────────────────────────
        self._build_ui()
        self.tip_lbl.configure(text="⏳  Loading DeepFilterNet3 model…",
                               text_color=COLOR_ACCENT)
        self.root.update_idletasks()

        self.dn = StreamingDenoiser(MODEL_DIR)
        log("model loaded")

        self.tip_lbl.configure(text="✓  Model loaded — select devices and press ON",
                               text_color=COLOR_ON)
        self._update_buttons()
        self._refresh_devices()
        self.root.after(UI_FPS_MS, self._tick)

    # ═════════════════════════════════════════════════════════════════════
    #  UI BUILDING
    # ═════════════════════════════════════════════════════════════════════


    def _build_ui(self):
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True)

        left = ctk.CTkFrame(main, width=220, fg_color=PANEL, corner_radius=0, border_width=1, border_color=PANEL_BORDER)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        center = ctk.CTkFrame(main, fg_color=BG, corner_radius=0)
        center.pack(side="left", fill="both", expand=True)

        right = ctk.CTkFrame(main, width=260, fg_color=PANEL, corner_radius=0, border_width=1, border_color=PANEL_BORDER)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_left_sidebar(left)
        self._build_center(center)
        self._build_right_sidebar(right)

        self.file_status = ctk.CTkLabel(main, text='')
        self.file_status.pack_forget()

    def _build_left_sidebar(self, parent):
        brand_fr = ctk.CTkFrame(parent, fg_color="transparent")
        brand_fr.pack(fill="x", pady=(20, 30), padx=15)
        ctk.CTkLabel(brand_fr, text="NeuroAcoustic", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(brand_fr, text="ENGINE · LIVE EDGE-COMPUTE", font=ctk.CTkFont(size=10), text_color=CYAN).pack(anchor="w")

        self._build_device_card(parent)
        
        ctk.CTkFrame(parent, fg_color="transparent").pack(fill="both", expand=True)
        
        self._build_network_card(parent)

    def _build_center(self, parent):
        top_fr = ctk.CTkFrame(parent, height=60, fg_color=PANEL, corner_radius=0, border_width=1, border_color=PANEL_BORDER)
        top_fr.pack(fill="x")
        top_fr.pack_propagate(False)

        self.onoff_btn = ctk.CTkButton(top_fr, text="SYSTEM MASTER", width=180, height=36,
                                       font=ctk.CTkFont(size=12, weight="bold"),
                                       fg_color=CRIMSON, hover_color=CRIMSON, corner_radius=18,
                                       command=self._toggle_main)
        self.onoff_btn.pack(side="left", padx=15, pady=12)

        self.repeat_btn = ctk.CTkButton(top_fr, text="LOOPBACK MONITOR", width=130, height=28,
                                        font=ctk.CTkFont(size=10, weight="bold"),
                                        fg_color=PANEL_BORDER, hover_color=TEXT_DIM,
                                        command=self._toggle_repeat)
        self.repeat_btn.pack(side="left", padx=5)

        self.rec_btn = ctk.CTkButton(top_fr, text="MIC TEST", width=90, height=28,
                                     font=ctk.CTkFont(size=10, weight="bold"),
                                     fg_color=PANEL_BORDER, hover_color=TEXT_DIM,
                                     command=self._toggle_record)
        self.rec_btn.pack(side="left", padx=5)

        ctk.CTkButton(top_fr, text="⚙ Settings", width=80, height=28,
                      font=ctk.CTkFont(size=10, weight="bold"), fg_color=PANEL_BORDER, hover_color=TEXT_DIM,
                      command=self._open_settings).pack(side="left", padx=15)
        
        self.stats_lbls = {}
        hdr_stats = ctk.CTkFrame(top_fr, fg_color="transparent")
        hdr_stats.pack(side="right", padx=15, pady=10)
        
        self.tip_lbl = ctk.CTkLabel(top_fr, text="", font=ctk.CTkFont(size=11), text_color=TEXT_DIM)
        self.tip_lbl.pack(side="left", fill="x", expand=True, padx=10)
        
        for key, name in [("lat", "LATENCY"), ("snr", "SNR"), ("infer", "INFERENCE")]:
            fr = ctk.CTkFrame(hdr_stats, fg_color="transparent")
            fr.pack(side="left", padx=8)
            ctk.CTkLabel(fr, text=name, font=ctk.CTkFont(size=9), text_color=TEXT_MUTED).pack()
            lbl = ctk.CTkLabel(fr, text="—", font=ctk.CTkFont(size=12, weight="bold"), text_color=CYAN)
            lbl.pack()
            self.stats_lbls[key] = lbl

        self._build_spectrograms_card(parent)
        self._build_stats_card(parent)
        
        hidden_fr = ctk.CTkFrame(parent)
        for k in ["rt", "blocks", "uptime", "params", "bins", "fft", "srio"]:
            l = ctk.CTkLabel(hidden_fr, text="")
            self.stats_lbls[k] = l

    def _build_right_sidebar(self, parent):
        self._build_voice_card(parent)

    def _build_device_card(self, parent):
        c = ctk.CTkFrame(parent, fg_color="transparent")
        c.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(c, text="🎤 INPUT SOURCE", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w")
        self.in_dev = ctk.CTkComboBox(c, width=190, font=ctk.CTkFont(size=11), command=self._save_config, fg_color=BG, border_color=PANEL_BORDER)
        self.in_dev.pack(pady=(2, 10))

        ctk.CTkLabel(c, text="🎧 MONITOR OUTPUT", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w")
        self.out_dev = ctk.CTkComboBox(c, width=190, font=ctk.CTkFont(size=11), fg_color=BG, border_color=PANEL_BORDER)
        self.out_dev.pack(pady=(2, 10))

        ctk.CTkLabel(c, text="⏱ BUFFER / LATENCY", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w")
        self.block_sel = ctk.CTkComboBox(c, width=190, values=["40", "100", "200", "400"], font=ctk.CTkFont(size=11), fg_color=BG, border_color=PANEL_BORDER)
        self.block_sel.set("100")
        self.block_sel.pack(pady=(2, 10))

    def _build_network_card(self, parent):
        c = ctk.CTkFrame(parent, fg_color="transparent")
        c.pack(fill="x", padx=15, pady=20)

        ctk.CTkLabel(c, text="NETWORK & TELEMETRY", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w", pady=(0,5))
        
        row1 = ctk.CTkFrame(c, fg_color="transparent")
        row1.pack(fill="x")
        self.net_status_lbl = ctk.CTkLabel(row1, text="● Disconnected", font=ctk.CTkFont(size=10), text_color=TEXT_DIM)
        self.net_status_lbl.pack(side="left")

        self.net_id_var = tk.StringVar(value="pc1")
        self.net_id_entry = ctk.CTkEntry(c, textvariable=self.net_id_var, width=190, height=28, font=ctk.CTkFont(size=11), fg_color=BG, border_color=PANEL_BORDER)
        self.net_id_entry.pack(pady=(5,5))

        self.net_connect_btn = ctk.CTkButton(c, text="CONNECT TO CLOUD", width=190, height=28, font=ctk.CTkFont(size=10, weight="bold"), fg_color=PANEL_BORDER, hover_color=TEXT_DIM, command=self._toggle_mqtt)
        self.net_connect_btn.pack()
        
        self.admin_override_lbl = ctk.CTkLabel(c, text="", font=ctk.CTkFont(size=9, weight="bold"), text_color=GREEN)
        self.admin_override_lbl.pack(anchor="w", pady=(5,0))

    def _build_spectrograms_card(self, parent):
        c = ctk.CTkFrame(parent, fg_color="transparent")
        c.pack(fill="both", expand=True, padx=20, pady=20)

        def make_spec(title, attr):
            fr = ctk.CTkFrame(c, fg_color="transparent")
            fr.pack(fill="both", expand=True, pady=5)
            
            lbl_fr = ctk.CTkFrame(fr, fg_color="transparent")
            lbl_fr.pack(anchor="w")
            ctk.CTkLabel(lbl_fr, text="●", font=ctk.CTkFont(size=10), text_color=GREEN).pack(side="left", padx=(0,5))
            ctk.CTkLabel(lbl_fr, text="LIVE", font=ctk.CTkFont(size=9, weight="bold"), text_color=GREEN).pack(side="left", padx=(0,10))
            ctk.CTkLabel(lbl_fr, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_PRIMARY).pack(side="left")

            ph = tk.PhotoImage(width=IMG_W, height=IMG_H)
            lbl = tk.Label(fr, image=ph, bg="black", borderwidth=1, highlightthickness=1, highlightbackground=PANEL_BORDER)
            lbl.image = ph
            lbl.pack(fill="both", expand=True, pady=(5, 0))
            setattr(self, attr, lbl)

        make_spec("RAW ACOUSTIC ENVIRONMENT", "spec_noisy")
        make_spec("NEURAL PROCESSED OUTPUT", "spec_enh")

    def _build_stats_card(self, parent):
        c = ctk.CTkFrame(parent, height=50, fg_color=PANEL, corner_radius=0, border_width=1, border_color=PANEL_BORDER)
        c.pack(fill="x", side="bottom")
        c.pack_propagate(False)

        inner = ctk.CTkFrame(c, fg_color="transparent")
        inner.pack(expand=True)

        items = [("ATTENUATION", "supp"), ("SPEECH", "spch"), ("INPUT", "in"), ("OUTPUT", "out"), ("DROPOUTS", "over"), ("MODEL", "model")]
        for i, (name, key) in enumerate(items):
            fr = ctk.CTkFrame(inner, fg_color="transparent")
            fr.pack(side="left", padx=15, pady=10)
            ctk.CTkLabel(fr, text=f"{name}:", font=ctk.CTkFont(size=9), text_color=TEXT_MUTED).pack(side="left", padx=(0,5))
            lbl = ctk.CTkLabel(fr, text="—", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_PRIMARY)
            lbl.pack(side="left")
            self.stats_lbls[key] = lbl

    def _build_voice_card(self, parent):
        c = ctk.CTkFrame(parent, fg_color="transparent")
        c.pack(fill="both", expand=True, padx=15, pady=20)

        ctk.CTkLabel(c, text="TARGET SPEAKER ENROLLMENT", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w")
        profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")
        has_profile = os.path.exists(profile_path)
        
        self.enroll_prompt = ctk.CTkLabel(c, text="VOICE PRINT ENROLLED" if has_profile else "No Voice Print", 
                                          font=ctk.CTkFont(size=10, weight="bold"), text_color=GREEN if has_profile else TEXT_DIM)
        self.enroll_prompt.pack(anchor="w", pady=(5, 10))

        self.enroll_btn = ctk.CTkButton(c, text="AUTHENTICATE VOICE PRINT", width=230, height=32,
                                        font=ctk.CTkFont(size=10, weight="bold"), fg_color=BG, border_color=CYAN, border_width=1,
                                        hover_color=PANEL_BORDER, command=self._enroll_voice)
        self.enroll_btn.pack()

        self.play_voice_btn = ctk.CTkButton(c, text="▶ LISTEN", width=230, height=28,
                                            font=ctk.CTkFont(size=10, weight="bold"), fg_color="transparent",
                                            state="normal" if has_profile else "disabled", command=self._play_voice)
        self.play_voice_btn.pack(pady=(5,20))

        ctk.CTkLabel(c, text="NEURAL EXTRACTION MODE", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w", pady=(0,10))
        self.mode_seg = ctk.CTkSegmentedButton(c, values=["BYPASS", "ZERO-LAT GATE", "DEEP EXTRACT"],
                                               font=ctk.CTkFont(size=9, weight="bold"), command=self._on_mode_change,
                                               selected_color=CYAN, selected_hover_color=CYAN, unselected_color=BG)
        self.mode_seg.pack(fill="x")
        self.mode_seg.set("BYPASS")
        self.mode_desc = ctk.CTkLabel(c, text="No speaker isolation active.", font=ctk.CTkFont(size=10), text_color=TEXT_DIM)
        self.mode_desc.pack(pady=(5,20))

        hidden_fr = ctk.CTkFrame(c)
        self.isolate_var = tk.BooleanVar(value=False)
        self.isolate_switch = ctk.CTkSwitch(hidden_fr, text="", variable=self.isolate_var)
        self.live_cocktail_var = tk.BooleanVar(value=False)
        self.live_cocktail_switch = ctk.CTkSwitch(hidden_fr, text="", variable=self.live_cocktail_var)
        
        ctk.CTkLabel(c, text="VOCAL SIGNATURE CONFIDENCE", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w")
        self.gauge_canvas = tk.Canvas(c, width=144, height=144, bg=PANEL, highlightthickness=0)
        self.gauge_canvas.pack(pady=10)
        self._draw_gauge(0.0)
        
        self.match_meter_frm = hidden_fr
        self.match_lbl = ctk.CTkLabel(hidden_fr, text="")
        self.match_bar = ctk.CTkProgressBar(hidden_fr)
        self.match_val_lbl = ctk.CTkLabel(hidden_fr, text="")

        thresh_fr = ctk.CTkFrame(c, fg_color="transparent")
        thresh_fr.pack(fill="x")
        ctk.CTkLabel(thresh_fr, text="THRESHOLD", font=ctk.CTkFont(size=9, weight="bold"), text_color=TEXT_DIM).pack(side="left")
        self.threshold_var = tk.DoubleVar(value=0.20)
        self.threshold_slider = ctk.CTkSlider(thresh_fr, from_=0.05, to=0.80, variable=self.threshold_var, command=self._on_thresh_change, width=100)
        self.threshold_slider.pack(side="left", padx=10)
        self.thresh_val_lbl = ctk.CTkLabel(thresh_fr, text="20%", font=ctk.CTkFont(size=9, weight="bold"), text_color=TEXT_PRIMARY)
        self.thresh_val_lbl.pack(side="left")

        ctk.CTkLabel(c, text="ROUTING", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w", pady=(20, 10))
        self.teams_var = tk.BooleanVar(value=False)
        self.teams_switch = ctk.CTkSwitch(c, text="Virtual Cable Injector", variable=self.teams_var,
                                          font=ctk.CTkFont(size=11), progress_color=CYAN, command=self._toggle_teams)
        self.teams_switch.pack(anchor="w")

    def _draw_gauge(self, score=0.0):
        canvas = self.gauge_canvas
        canvas.delete('all')
        cx, cy, r = 72, 72, 58
        thickness = 12
        
        canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=-225, extent=270, style=tk.ARC, width=thickness, outline=PANEL_BORDER)
        
        extent = 270 * score
        if score < 0.3:
            color = CRIMSON
            lbl_text = "MUTED"
        elif score < 0.6:
            color = "yellow"
            lbl_text = "WEAK"
        elif score < 0.8:
            color = GREEN
            lbl_text = "MATCH"
        else:
            color = CYAN
            lbl_text = "MATCH"
            
        if extent > 0:
            canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=-225, extent=extent, style=tk.ARC, width=thickness, outline=color)
            
        canvas.create_text(cx, cy - 10, text=f"{int(score*100)}%", font=("Arial", 20, "bold"), fill=TEXT_PRIMARY)
        canvas.create_text(cx, cy + 15, text=lbl_text, font=("Arial", 9, "bold"), fill=color)

    def _on_mode_change(self, value):
        if value == 'BYPASS':
            self.mode_desc.configure(text="No speaker isolation active.")
            if self.isolate_var.get():
                self.isolate_var.set(False)
                self._toggle_isolate()
            if self.live_cocktail_var.get():
                self.live_cocktail_var.set(False)
                self._toggle_cocktail()
        elif value == 'ZERO-LAT GATE':
            self.mode_desc.configure(text="Fast biometric gating. Low latency.")
            if self.live_cocktail_var.get():
                self.live_cocktail_var.set(False)
                self._toggle_cocktail()
            if not self.isolate_var.get():
                self.isolate_var.set(True)
                self._toggle_isolate()
        elif value == 'DEEP EXTRACT':
            self.mode_desc.configure(text="Deep neural separation. 3s delay.")
            if self.isolate_var.get():
                self.isolate_var.set(False)
                self._toggle_isolate()
            if not self.live_cocktail_var.get():
                self.live_cocktail_var.set(True)
                self._toggle_cocktail()

    def _open_settings(self):
        win = ctk.CTkToplevel(self.root)
        win.title("Audio Controls")
        win.geometry("500x550")
        win.attributes("-topmost", True)
        win.configure(fg_color=BG)
        
        c = ctk.CTkFrame(win, fg_color="transparent")
        c.pack(fill="both", expand=True, padx=20, pady=20)
        c.columnconfigure(1, weight=1)

        ctk.CTkLabel(c, text="Environment Profile:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w", pady=10)
        if not hasattr(self, "profile_var"):
            self.profile_var = tk.StringVar(value="Custom")
        self.profile_sel = ctk.CTkComboBox(c, variable=self.profile_var, values=["Custom", "Traffic (Max Suppression)", "Classroom (Babble Control)", "Home (Natural Mix)"], command=self._on_profile_change, width=250, state="readonly")
        self.profile_sel.grid(row=0, column=1, columnspan=2, sticky="ew", pady=10)
        
        if not hasattr(self, "fx_lbls_live"):
            self.fx_lbls_live = {}
        if not hasattr(self, "fx_lbls_file"):
            self.fx_lbls_file = {}
            
        cur_row = 1
        
        ctk.CTkLabel(c, text="Noise Attenuation", font=ctk.CTkFont(size=12, weight="bold")).grid(row=cur_row, column=0, sticky="w", pady=5)
        if not hasattr(self, "atten_var"):
            self.atten_var = tk.DoubleVar(value=self.params["live"]["atten"])
        atten_s = ctk.CTkSlider(c, from_=0, to=100, variable=self.atten_var, command=lambda v: self._on_param("live", "atten", v))
        atten_s.grid(row=cur_row, column=1, sticky="ew", padx=10, pady=5)
        
        self.atten_lbl = ctk.CTkLabel(c, text=f'{int(self.atten_var.get())} dB', font=ctk.CTkFont(size=12, weight="bold"), width=60)
        self.atten_lbl.grid(row=cur_row, column=2, pady=5)
        cur_row += 1
        
        def _slider(row, name, label, frm, to, init, fmt):
            ctk.CTkLabel(c, text=label, font=ctk.CTkFont(size=11)).grid(row=row, column=0, sticky="w", pady=5)
            if name in self.fx_lbls_live:
                var = self.fx_lbls_live[name][0]
            else:
                var = tk.DoubleVar(value=init)
                
            s = ctk.CTkSlider(c, from_=frm, to=to, variable=var, command=lambda v, n=name: self._on_param("live", n, v))
            s.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
            
            lbl = ctk.CTkLabel(c, text=fmt(var.get()), font=ctk.CTkFont(size=11, weight="bold"), width=60)
            lbl.grid(row=row, column=2, pady=5)
            
            self.fx_lbls_live[name] = (var, lbl, fmt)
            self.fx_lbls_file[name] = (var, lbl, fmt)
            
        p = self.params["live"]
        _slider(cur_row, "mix", "Dry / Wet Mix", 0, 100, p["mix"], lambda v: f"{round(v)} %")
        cur_row += 1
        _slider(cur_row, "gain", "Output Gain", -12, 12, p["gain"], lambda v: f"{round(v):+d} dB")
        cur_row += 1
        _slider(cur_row, "hpf", "High-Pass Filter", 0, 400, p["hpf"], lambda v: "off" if round(v / 20) * 20 < 20 else f"{round(v / 20) * 20} Hz")
        cur_row += 1
        _slider(cur_row, "floor", "Max Suppression", 0, 100, p["floor"], lambda v: "off" if round(v) >= 100 else f"{round(v)} dB")
        cur_row += 1
        
        if not hasattr(self, "pf_var"):
            self.pf_var = tk.BooleanVar(value=self.pf)
        self.pf_cb = ctk.CTkCheckBox(c, text="Aggressive post-filter (reloads model)", variable=self.pf_var, command=self._on_pf)
        self.pf_cb.grid(row=cur_row, column=0, columnspan=3, sticky="w", pady=(20, 5))
        cur_row += 1
        
        if not hasattr(self, "siren_var"):
            self.siren_var = tk.BooleanVar(value=False)
        self.siren_cb = ctk.CTkCheckBox(c, text="Siren and Whistle Killer", variable=self.siren_var, command=self._on_siren)
        self.siren_cb.grid(row=cur_row, column=0, columnspan=3, sticky="w", pady=5)

    def _update_buttons(self):
        denoise_on = self.mode == "denoise"
        self.onoff_btn.configure(
            text="SYSTEM MASTER (ON)" if denoise_on else "SYSTEM MASTER",
            fg_color=GREEN if denoise_on else CRIMSON,
            hover_color=GREEN if denoise_on else CRIMSON)
            
        if self.repeat:
            self.repeat_btn.configure(fg_color=CYAN, text_color=BG)
        else:
            self.repeat_btn.configure(fg_color=PANEL_BORDER, text_color=TEXT_PRIMARY)
                
        if self.mode == "bypass":
            self.tip_lbl.configure(text="ℹ BYPASS MODE: Playing raw audio", text_color="yellow")
        elif self.mode == "denoise":
            self.tip_lbl.configure(text="⚡ NEURAL DENOISING ACTIVE", text_color=GREEN)
        else:
            self.tip_lbl.configure(text="")

    # ═════════════════════════════════════════════════════════════════════
    #  DEVICE MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════


    def _toggle_mqtt(self):
        if self.remote_controller is None:
            # Connect
            dev_id = self.net_id_var.get().strip()
            if not dev_id:
                return
            from remote_controller import RemoteController
            self.remote_controller = RemoteController(dev_id, self._on_mqtt_command, self._on_mqtt_status)
            self.remote_controller.start()
            self._broadcast_state()
            self.net_connect_btn.configure(text="Disconnect", fg_color=COLOR_OFF, hover_color=COLOR_OFF_HOVER)
        else:
            # Disconnect
            self.remote_controller.stop()
            self.remote_controller = None
            self.net_connect_btn.configure(text="Connect to Cloud", fg_color=COLOR_ACCENT, hover_color=COLOR_ON_HOVER)
            self.net_status_lbl.configure(text="Disconnected", text_color=COLOR_DIM)
            self.admin_override_lbl.configure(text="")

    def _broadcast_state(self):
        if getattr(self, "remote_controller", None) is not None:
            self.remote_controller.publish_state(
                role=self.profile_var.get(),
                denoise_on=(self.mode == "denoise"),
                isolate_on=self.isolate_var.get()
            )

    def _on_mqtt_command(self, action, state):
        self.root.after(0, lambda: self._handle_admin_override(action, state))

    def _on_mqtt_status(self, status_msg, is_error):
        color = COLOR_OFF if is_error else COLOR_ON
        self.root.after(0, lambda: self.net_status_lbl.configure(text=status_msg, text_color=color))

    def _handle_admin_override(self, action, state):
        self.admin_override_lbl.configure(text=f"Controlled by Admin: {action.upper()} -> {state}")
        
        if action == "denoise":
            current = (self.mode == "denoise")
            if current != state:
                self._toggle_main()
        elif action == "isolate":
            current = self.isolate_var.get()
            if current != state:
                self.isolate_var.set(state)
                self._toggle_isolate()
        elif action == "profile":
            current = self.profile_var.get()
            if current != state and hasattr(self, "profile_var"):
                self.profile_var.set(state)
                self._on_profile_change(state)



    def _toggle_isolate(self):
        if self.isolate_var.get() and getattr(self, "live_cocktail_var", None) and self.live_cocktail_var.get():
            self.live_cocktail_var.set(False)
            self._toggle_cocktail()
            
        if self.dn is not None:
            self.dn.isolate_speaker = self.isolate_var.get()
            if self.isolate_var.get():
                if hasattr(self, "threshold_var"):
                    self.threshold_var.set(0.20)
                    self._on_thresh_change(0.20)
                self.dn.reload_profile()
        self._broadcast_state()

    def _toggle_teams(self):
        is_teams = self.teams_var.get()
        if is_teams:
            outs = self.out_dev._values
            match = [t for t in outs if "CABLE Input" in t or "Virtual" in t]
            if match:
                self._prev_out_dev = self.out_dev.get()
                self.out_dev.set(match[0])
                self.tip_lbl.configure(text="Teams Mode ON: AI audio is now routed to Virtual Cable.", text_color=COLOR_ON)
            else:
                self.teams_var.set(False)
                self.tip_lbl.configure(text="VB-Audio Virtual Cable not found! Please install it.", text_color="red")
        else:
            if hasattr(self, "_prev_out_dev"):
                self.out_dev.set(self._prev_out_dev)
            self.tip_lbl.configure(text="Teams Mode OFF: Audio routed back to normal speakers.", text_color=COLOR_ON)
            
        if self.running:
            self._stop_all()
            self.root.after(100, self._start_all)

    def _toggle_cocktail(self):
        if self.live_cocktail_var.get() and getattr(self, "isolate_var", None) and self.isolate_var.get():
            self.isolate_var.set(False)
            self._toggle_isolate()
            
        if self.dn is not None:
            enabled = self.live_cocktail_var.get()
            if enabled:
                if hasattr(self, "threshold_var"):
                    self.threshold_var.set(0.15)
                    self._on_thresh_change(0.15)
                self.live_cocktail_switch.configure(text="Loading Cocktail AI...", state="disabled")
                self.tip_lbl.configure(text="Loading massive SepFormer AI into memory (takes a few secs)...", text_color="yellow")
                self.root.update_idletasks()
                
                def load_task():
                    try:
                        self.dn.set_cocktail_mode(True)
                        self.root.after(0, lambda: self.live_cocktail_switch.configure(text="Cocktail Party Separation (3s Delay)", state="normal"))
                        self.root.after(0, lambda: self.tip_lbl.configure(text="Cocktail AI Active! (Expect 3s audio delay)", text_color=COLOR_ON))
                    except Exception as e:
                        log(f"Failed to load cocktail mode: {e!r}")
                        self.root.after(0, lambda: self.live_cocktail_var.set(False))
                        self.root.after(0, lambda: self.live_cocktail_switch.configure(text="Cocktail Party Separation (3s Delay)", state="normal"))
                
                import threading
                threading.Thread(target=load_task, daemon=True).start()
            else:
                self.dn.set_cocktail_mode(False)
                self.tip_lbl.configure(text="Cocktail Mode OFF. Standard Denoising Active.", text_color=COLOR_ON)

    def _play_voice(self):
        profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")
        if not os.path.exists(profile_path):
            return
            
        try:
            dev_out = int(self.out_dev.get().split(":")[0])
        except (ValueError, IndexError):
            self.enroll_prompt.configure(text="Please select an output device first!")
            return
            
        self.play_voice_btn.configure(state="disabled")
        self.enroll_prompt.configure(text="Playing your saved Voice Print...", text_color=COLOR_ON)
        
        def play_task():
            try:
                import soundfile as sf
                data, fs = sf.read(profile_path)
                sd.play(data, fs, device=dev_out)
                sd.wait()
                self.enroll_prompt.configure(text="Finished playing back your Voice Print.", text_color=COLOR_DIM)
            except Exception as e:
                self.enroll_prompt.configure(text=f"Error playing: {e}", text_color=COLOR_OFF)
            finally:
                self.play_voice_btn.configure(state="normal")
                
        threading.Thread(target=play_task, daemon=True).start()

    def _enroll_voice(self):
        try:
            dev_in = int(self.in_dev.get().split(":")[0])
        except (ValueError, IndexError):
            self.enroll_prompt.configure(text="Please select a microphone first!")
            return

        self.enroll_btn.configure(state="disabled")
        self.play_voice_btn.configure(state="disabled")
        self.enroll_prompt.configure(text="RECORDING (20s)...", text_color="orange")
        
        abort_flag = [False]
        
        # --- Teleprompter Overlay ---
        prompt_win = ctk.CTkToplevel(self.root)
        
        def on_close_prompt():
            abort_flag[0] = True
            import sounddevice as sd
            sd.stop()
            prompt_win.destroy()
            self.enroll_prompt.configure(text="Enrollment aborted by user.", text_color="#FF4444")
            self.enroll_btn.configure(state="normal", text="Re-enroll Voice")
            if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")):
                self.play_voice_btn.configure(state="normal")
                self.isolate_switch.configure(state="normal")

        prompt_win.protocol("WM_DELETE_WINDOW", on_close_prompt)
        prompt_win.title("Voice Enrollment")
        prompt_win.geometry("500x350")
        prompt_win.attributes("-topmost", True)
        prompt_win.geometry(f"+{self.root.winfo_x() + 100}+{self.root.winfo_y() + 50}")
        
        lbl_title = ctk.CTkLabel(prompt_win, text="RECORDING IN PROGRESS", text_color="#FF4444", font=ctk.CTkFont(size=20, weight="bold"))
        lbl_title.pack(pady=(20, 10))
        
        script_text = (
            "Please read the following text in your normal voice:\n\n"
            "\"Hello! I am recording my voice to create a biometric profile for my Advanced Machine Learning project. "
            "The system is currently mapping the unique frequencies of my vocal cords. "
            "Once I finish reading this paragraph, the AI will use this recording to mathematically separate my voice from any other people talking in the background. "
            "This is a live test of the Cocktail Party separation system!\""
        )
        lbl_script = ctk.CTkLabel(prompt_win, text=script_text, font=ctk.CTkFont(size=15), wraplength=450, justify="center")
        lbl_script.pack(pady=10, padx=20)
        
        lbl_timer = ctk.CTkLabel(prompt_win, text="20 seconds remaining...", font=ctk.CTkFont(size=18, weight="bold"))
        lbl_timer.pack(pady=(20, 20))
        
        def update_timer(secs):
            if not prompt_win.winfo_exists():
                return
            if secs > 0:
                lbl_timer.configure(text=f"{secs} seconds remaining...")
                self.root.after(1000, update_timer, secs - 1)
            else:
                lbl_timer.configure(text="Purifying audio (DeepFilterNet)...", text_color="yellow")
        
        update_timer(20)
        # ----------------------------

        def record_task():
            try:
                import soundfile as sf
                audio = sd.rec(int(20 * 48000), samplerate=48000, channels=1, device=dev_in, dtype='float32')
                sd.wait()
                
                if abort_flag[0]:
                    return  # User closed the window, abort the save process completely
                
                profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")
                temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile_raw.wav")
                
                # Save the raw noisy mic recording
                sf.write(temp_path, audio, 48000)
                
                self.enroll_prompt.configure(text="Purifying voice print using DeepFilterNet...", text_color="yellow")
                if prompt_win.winfo_exists():
                    prompt_win.destroy()
                
                # Denoise the profile offline before ECAPA sees it!
                if self.dn is not None:
                    self.dn.denoise_file(temp_path, profile_path)
                else:
                    sf.write(profile_path, audio, 48000)
                    
                # --- Smart Silence Trimming (Voice Activity Detection) ---
                import numpy as np
                clean_audio, sr = sf.read(profile_path)
                
                # Find the first moment of actual speech (skipping up to 5 seconds of silence)
                threshold = 0.01  # RMS threshold
                window = int(sr * 0.1) # 100ms
                start_idx = 0
                # Ignore first 0.4 seconds to bypass DeepFilterNet startup clicks
                for i in range(int(sr * 0.4), len(clean_audio), window):
                    chunk = clean_audio[i:i+window]
                    if np.sqrt(np.mean(chunk**2)) > threshold:
                        start_idx = max(0, i - int(sr * 0.15)) # 150ms pre-roll to keep breath/start of word
                        break
                        
                # Crop EXACTLY 15 seconds starting from the first spoken word!
                end_idx = min(len(clean_audio), start_idx + int(15 * sr))
                final_15s_audio = clean_audio[start_idx:end_idx]
                sf.write(profile_path, final_15s_audio, sr)
                # ---------------------------------------------------------
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                self.enroll_prompt.configure(text="Voice Print saved! The AI will now use this to filter out other voices.", text_color=COLOR_ON)
                self.enroll_btn.configure(text="Re-enroll Voice")
            except Exception as e:
                self.enroll_prompt.configure(text=f"Error recording: {e}", text_color=COLOR_OFF)
            finally:
                self.enroll_btn.configure(state="normal")
                if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")):
                    self.play_voice_btn.configure(state="normal")
                    self.isolate_switch.configure(state="normal")
                
                
        threading.Thread(target=record_task, daemon=True).start()


    def _save_config(self, _=None):
        in_str = self.in_dev.get()
        out_str = self.out_dev.get()
        # Strip the index number (e.g. "1: CABLE" -> "CABLE")
        in_name = in_str.split(": ", 1)[1] if ": " in in_str else in_str
        out_name = out_str.split(": ", 1)[1] if ": " in out_str else out_str
        
        self.saved_config["in_dev_name"] = in_name
        self.saved_config["out_dev_name"] = out_name
        
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.saved_config, f)
        except Exception as e:
            print("Failed to save config:", e)

    def _refresh_devices(self):
        devs = sd.query_devices()
        default_api = sd.default.hostapi
        ins, outs = [], []
        for i, d in enumerate(devs):
            if d['hostapi'] != default_api:
                continue
            if d["max_input_channels"] > 0:
                ins.append(f"{i}: {d['name']}")
            if d["max_output_channels"] > 0:
                outs.append(f"{i}: {d['name']}")
        if not ins:
            self.tip_lbl.configure(text="⚠  No input audio devices found.",
                                   text_color=COLOR_OFF)
        if not outs:
            self.tip_lbl.configure(text="⚠  No output audio devices found.",
                                   text_color=COLOR_OFF)
        self.in_dev.configure(values=ins)
        self.out_dev.configure(values=outs)
        din, dout = sd.default.device
        # Load saved devices based on name (ignoring indices which can change)
        saved_in = self.saved_config.get("in_dev_name", "")
        saved_out = self.saved_config.get("out_dev_name", "")
        
        in_match = [t for t in ins if saved_in in t] if saved_in else []
        if in_match:
            self.in_dev.set(in_match[0])
        elif 0 <= din < len(devs) and ins:
            match = [t for t in ins if t.startswith(f"{din}:")]
            self.in_dev.set(match[0] if match else ins[0])
            
        out_match = [t for t in outs if saved_out in t] if saved_out else []
        if out_match:
            self.out_dev.set(out_match[0])
        elif 0 <= dout < len(devs) and outs:
            match = [t for t in outs if t.startswith(f"{dout}:")]
            self.out_dev.set(match[0] if match else outs[0])

    # ═════════════════════════════════════════════════════════════════════
    #  PARAMETERS
    # ═════════════════════════════════════════════════════════════════════

    def _on_thresh_change(self, val):
        self.thresh_val_lbl.configure(text=f"{int(val*100)}%")
        if self.dn:
            self.dn.biometric_threshold = float(val)

    def _on_profile_change(self, choice: str):
        if choice == "Custom":
            self._broadcast_state()
            return
        
        # Define the presets: (atten, mix, gain, hpf, floor, post_filter)
        presets = {
            "Traffic (Max Suppression)": (100, 100, 12, 160, 100, False),
            "Classroom (Babble Control)": (100, 100, 12, 100, 100, True),
            "Home (Natural Mix)": (60, 95, 12, 60, 40, False)
        }
        
        if choice in presets:
            atten, mix, gain, hpf, floor, pf = presets[choice]
            
            self._suppress_custom = True
            
            # Update Sliders & Values
            self.atten_var.set(atten)
            self._on_param("live", "atten", atten)
            
            self.fx_lbls_live["mix"][0].set(mix)
            self._on_param("live", "mix", mix)
            
            self.fx_lbls_live["gain"][0].set(gain)
            self._on_param("live", "gain", gain)
            
            self.fx_lbls_live["hpf"][0].set(hpf)
            self._on_param("live", "hpf", hpf)
            
            self.fx_lbls_live["floor"][0].set(floor)
            self._on_param("live", "floor", floor)
            
            # Update Post Filter
            self.pf_var.set(pf)
            self._on_pf()
            
            self._suppress_custom = False
            self._broadcast_state()


    def _on_param(self, tab, name, raw):
        """Slider moved: store value, update label, apply to pipeline."""
        if tab == "live" and not getattr(self, "_suppress_custom", False):
            if hasattr(self, "profile_var") and self.profile_var.get() != "Custom":
                self.profile_var.set("Custom")
                self._broadcast_state()
                
        if name == "atten":
            v = int(round(float(raw)))
            self.params[tab]["atten"] = v
            if tab == "live":
                self.atten_lbl.configure(text=f"{v} dB")
                if self.mode == "denoise":
                    self.dn.set_atten_lim(v)
            return
        var, lbl, fmt = getattr(self, f"fx_lbls_{tab}")[name]
        if name == "mix":
            v = int(round(float(raw)))
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_mix(v / 100.0)
        elif name == "gain":
            v = round(float(raw))
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_output_gain(v)
        elif name == "hpf":
            v = round(float(raw) / 20) * 20
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_highpass(v)
        elif name == "floor":
            v = round(float(raw))
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_mask_floor(v)
        self.params[tab][name] = v

    def _apply_params(self, tab):
        p = self.params[tab]
        self.dn.set_atten_lim(p["atten"])
        self.dn.set_mix(p["mix"] / 100.0)
        self.dn.set_output_gain(p["gain"])
        self.dn.set_highpass(p["hpf"])
        self.dn.set_mask_floor(p["floor"])
        if hasattr(self, "threshold_var"):
            self.dn.biometric_threshold = float(self.threshold_var.get())
        if hasattr(self, "isolate_var"):
            self.dn.isolate_speaker = self.isolate_var.get()

    def _on_pf(self):
        if not getattr(self, "_suppress_custom", False):
            if hasattr(self, "profile_var") and self.profile_var.get() != "Custom":
                self.profile_var.set("Custom")
                
        if self.running:
            self.tip_lbl.configure(
                text="ℹ  Post-filter change applies next time you turn OFF → ON.",
                text_color=COLOR_DIM)
            return
        self._ensure_denoiser()

    def _on_siren(self):
        """Toggle the custom Spectral Median Siren filter."""
        enabled = self.siren_var.get()
        if hasattr(self, "dn") and self.dn is not None:
            self.dn.set_siren_filter(enabled)
            log(f"Siren Killer set to {enabled}")

    def _toggle_record(self):
        """Toggle recording of raw mic + AI output to WAV files."""
        if not getattr(self, '_recording', False):
            # START recording
            if not self.running:
                self.tip_lbl.configure(
                    text="⚠  Turn ON the denoiser first, then hit Record.",
                    text_color=COLOR_OFF)
                return
            self._rec_raw.clear()
            self._rec_ai.clear()
            self._recording = True
            self.rec_btn.configure(text="⏹ Stop Rec", fg_color="#B22222")
            self.tip_lbl.configure(text="🔴  Recording... speak now, then click Stop Rec.",
                                   text_color="#FF4444")
            log("Recording started")
        else:
            # STOP recording and save files
            self._recording = False
            self.rec_btn.configure(text="🔴 Record", fg_color="gray25")
            if not self._rec_raw:
                self.tip_lbl.configure(text="⚠  Nothing was recorded.",
                                       text_color=COLOR_OFF)
                return
            import soundfile as sf_lib
            os.makedirs(self.out_dir, exist_ok=True)
            raw = np.concatenate(self._rec_raw)
            ai = np.concatenate(self._rec_ai)
            # Align lengths (crossfade may cause slight difference)
            n = min(len(raw), len(ai))
            raw, ai = raw[:n], ai[:n]
            raw_path = os.path.join(self.out_dir, "recorded_raw.wav")
            ai_path = os.path.join(self.out_dir, "recorded_ai.wav")
            sf_lib.write(raw_path, raw, self.dn.sr)
            sf_lib.write(ai_path, ai, self.dn.sr)
            dur = n / self.dn.sr
            self.tip_lbl.configure(
                text=f"✓  Saved {dur:.1f}s → recorded_raw.wav + recorded_ai.wav in output/",
                text_color=COLOR_ON)
            log(f"Recording saved: {dur:.1f}s  raw={raw_path}  ai={ai_path}")
            self._rec_raw.clear()
            self._rec_ai.clear()

    def _ensure_denoiser(self):
        """Rebuild the denoiser if the post-filter option changed."""
        pf = bool(self.pf_var.get())
        if self.pf == pf:
            return
        self.tip_lbl.configure(text="⏳  Reloading model…", text_color=COLOR_ACCENT)
        self.root.update_idletasks()
        self.dn = StreamingDenoiser(MODEL_DIR, post_filter=pf)
        self.pf = pf
        log(f"model reloaded with post_filter={pf}")
        msg = f"✓  Model reloaded (post-filter {'ON' if pf else 'OFF'})"
        self.tip_lbl.configure(text=msg, text_color=COLOR_ON)
        self.file_status.configure(text=msg, text_color=COLOR_ON)
        
        # Sync states
        if hasattr(self, 'isolate_var'):
            self.dn.isolate_speaker = self.isolate_var.get()
        if hasattr(self, 'live_cocktail_var'):
            self.dn.set_cocktail_mode(self.live_cocktail_var.get())

    # ═════════════════════════════════════════════════════════════════════
    #  ENGINE CONTROL
    # ═════════════════════════════════════════════════════════════════════

    # state matrix (main ON/OFF, Repeat ON/OFF):
    #   OFF/OFF  nothing runs
    #   OFF/ON   raw passthrough: mic -> speaker, unfiltered (bypass mode)
    #   ON /ON   denoised audio played to output
    #   ON /OFF  denoiser runs (stats, spectrograms) but output stays silent

    def _toggle_main(self):
        was_denoise = (self.mode == "denoise")
        self._stop_all()
        
        if not was_denoise:
            # They want to turn denoising ON
            self._start_engine(denoise=True)
        else:
            # They want to turn denoising OFF.
            # We ALWAYS fall back to raw bypass mode so Teams still gets normal audio!
            self._start_engine(denoise=False)
            
        self._broadcast_state()

    def _toggle_repeat(self):
        self.repeat = not self.repeat
        was_mode = self.mode
        self._stop_all()
        
        if was_mode == "denoise":
            # Keep denoising, just with/without output
            self._start_engine(denoise=True)
        else:
            # If engine was stopped (None) and repeat turned ON -> Start bypass mode!
            # If engine was in bypass and repeat turned OFF -> Both are OFF, do not start.
            if self.repeat:
                self._start_engine(denoise=False)
                
        self._update_buttons()


    def _start_engine(self, denoise: bool = True):
        self._ensure_denoiser()
        self.block_ms = int(self.block_sel.get())
        self.block_n = self.dn.hop * self.block_ms // 10
        if denoise:
            self.dn.reset()
            self._apply_params("live")
        self.overruns = 0
        self.start_time = time.perf_counter()

        try:
            dev_in = int(self.in_dev.get().split(":")[0])
            dev_out = int(self.out_dev.get().split(":")[0])
        except (ValueError, IndexError):
            self.tip_lbl.configure(
                text="⚠  Please select both a microphone and an output device.",
                text_color=COLOR_OFF)
            return

        def in_cb(indata, frames, t, status):
            if status:
                log(f"input stream status: {status}")
            try:
                self.in_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                self.overruns += 1

        def out_cb(outdata, frames, t, status):
            with self.jb_lock:
                outdata[:, 0] = self.jb.read(frames)

        try:
            self.in_stream = sd.InputStream(
                samplerate=self.dn.sr, channels=1, dtype="float32",
                blocksize=self.block_n, device=dev_in, callback=in_cb)
            if self.repeat:
                self.out_stream = sd.OutputStream(
                    samplerate=self.dn.sr, channels=1, dtype="float32",
                    blocksize=self.block_n, device=dev_out, callback=out_cb)
        except Exception as e:
            err_msg = f'Could not open audio device: {e}\n\nDid you select the correct Microphone and Output?'
            self.tip_lbl.configure(text=err_msg, text_color=COLOR_OFF)
            log(f'ERROR opening streams: {e}')
            import tkinter.messagebox
            tkinter.messagebox.showerror('Audio Error', err_msg)
            return

        log(f"stream opened: mode={'denoise' if denoise else 'bypass'} "
            f"repeat={self.repeat} in={self.in_dev.get()} out={self.out_dev.get()} "
            f"block={self.block_ms}ms sr={self.dn.sr}")
        self._silent_s = 0
        # ~3 blocks of backlog: absorbs clock drift + PyTorch thread jitter
        # Enforce a minimum of 4800 samples (100ms) safety net to completely eliminate "kr kr" tearing on Windows.
        target = max(int(self.block_n * 3.0), 4800)
        self.jb = JitterBuffer(target_samples=target)
        self.in_stream.start()
        if self.out_stream:
            self.out_stream.start()
        self.mode = "denoise" if denoise else "bypass"
        self.running = True   # must be set before the worker starts its loop
        self.worker = threading.Thread(target=self._process_loop, daemon=True)
        self.worker.start()
        self._update_buttons()

    def _stop_all(self):
        self.running = False
        self.mode = None
        if self.worker:
            self.worker.join(timeout=2)
            self.worker = None
        for s in (self.in_stream, self.out_stream):
            if s:
                s.stop()
                s.close()
        self.in_stream = self.out_stream = None
        with self.jb_lock:
            if self.jb:
                self.jb.clear()
        self._update_buttons()
        log("streams stopped")

    def _process_loop(self):
        _diag_count = 0
        while self.running:
            try:
                blk = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            
            _diag_count += 1
            if _diag_count % 50 == 1:
                peak = float(np.max(np.abs(blk)))
                log(f"DIAG: block #{_diag_count}, peak={peak:.6f}, len={len(blk)}")
                
            try:
                if self.mode == "bypass":
                    self.dn.process_bypass(blk)
                    out = blk
                else:
                    out = self.dn.process_block(blk)
            except Exception:
                log("worker crash:\n" + traceback.format_exc())
                self.running = False
                self.root.after(0, self._on_worker_crash)
                return
            if self.repeat:
                with self.jb_lock:
                    self.jb.append(out)
            if getattr(self, "_recording", False):
                self._rec_raw.append(blk)
                self._rec_ai.append(out)

    def _on_worker_crash(self):
        """Called on the main thread when the audio worker thread dies."""
        self._stop_all()
        self.tip_lbl.configure(
            text="⚠  Audio processing error — check audiodenoiser.log",
            text_color=COLOR_OFF)

    # ═════════════════════════════════════════════════════════════════════
    #  FILE MODE
    # ═════════════════════════════════════════════════════════════════════

    def _add_files(self):
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title="Choose audio files to denoise",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma"),
                ("All files", "*.*")])
        existing = set(self.file_list.get(0, "end"))
        for p in paths:
            if p not in existing:
                self.file_list.insert("end", p)

    def _remove_files(self):
        for idx in sorted(self.file_list.curselection(), reverse=True):
            self.file_list.delete(idx)

    def _clear_files(self):
        self.file_list.delete(0, "end")

    def _open_out_dir(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.startfile(self.out_dir)

    def _denoise_files(self):
        if self._file_busy:
            return
        if self.mode is not None:
            self.file_status.configure(
                text="⚠  Turn the live denoiser OFF before processing files.",
                text_color=COLOR_OFF)
            return
        files = list(self.file_list.get(0, "end"))
        if not files:
            self.file_status.configure(text="⚠  Add some files first.",
                                       text_color=COLOR_OFF)
            return
        self._ensure_denoiser()
        self._file_busy = True
        self.denoise_btn.configure(state="disabled")
        self.onoff_btn.configure(state="disabled")     # prevent live during file
        os.makedirs(self.out_dir, exist_ok=True)

        def worker():
            results = []
            do_separate = self.file_separate_var.get()
            
            if do_separate:
                try:
                    import torch
                    from multi_speaker_separator import MultiSpeakerSeparator
                    import soundfile as sf
                    self._file_progress = "Loading massive Separation AI (this takes a moment)..."
                    separator = MultiSpeakerSeparator(device="cuda" if torch.cuda.is_available() else "cpu")
                    profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")
                except Exception as e:
                    log(f"Failed to load separator: {e!r}")
                    self._file_results = [("fail", (files[0], str(e)))]
                    self._file_done = True
                    return

            for i, src in enumerate(files):
                self._file_progress = f"Processing {i + 1}/{len(files)}: {os.path.basename(src)}"
                try:
                    stem = os.path.splitext(os.path.basename(src))[0]
                    dst = os.path.join(self.out_dir, f"{stem}_denoised.wav")
                    
                    if do_separate:
                        def progress_cb(prog):
                            self._file_progress = f"Separating Overlapping Voices: {int(prog*100)}%"
                            
                        final_audio, sr = separator.separate_and_isolate(
                            src, profile_path, progress_callback=progress_cb
                        )
                        
                        # Save the separated audio to a temp file, then denoise it to remove robotic artifacts
                        temp_dst = os.path.join(self.out_dir, f"{stem}_temp.wav")
                        sf.write(temp_dst, final_audio.squeeze(0).numpy(), sr)
                        
                        self._file_progress = f"Denoising {os.path.basename(src)}..."
                        self._apply_params("file")
                        r = self.dn.denoise_file(temp_dst, dst)
                        if os.path.exists(temp_dst):
                            os.remove(temp_dst)
                    else:
                        self._apply_params("file")
                        r = self.dn.denoise_file(src, dst)
                        
                    results.append(("ok", r))
                except Exception as e:
                    log(f"file denoise failed for {src}: {e!r}")
                    results.append(("fail", (src, str(e))))
            self._file_results = results
            self._file_done = True

        self._file_progress = "Starting…"
        self._file_done = False
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(200, self._poll_file_worker)

    def _poll_file_worker(self):
        self.file_status.configure(text=f"⏳  {self._file_progress}",
                                   text_color=COLOR_ACCENT)
        if not self._file_done:
            self.root.after(200, self._poll_file_worker)
            return
        self._file_busy = False
        self.denoise_btn.configure(state="normal")
        self.onoff_btn.configure(state="normal")       # re-enable live button
        ok = [r for s, r in self._file_results if s == "ok"]
        fail = [r for s, r in self._file_results if s == "fail"]
        msg = f"✓  Done: {len(ok)} file(s) denoised → {self.out_dir}"
        if fail:
            msg += f"  ({len(fail)} failed — see audiodenoiser.log)"
        self.file_status.configure(
            text=msg, text_color=COLOR_OFF if fail else COLOR_ON)
        log(msg)

    # ═════════════════════════════════════════════════════════════════════
    #  TICK / UPDATES
    # ═════════════════════════════════════════════════════════════════════

    def _tick(self):
        try:
            self._update_stats()
            self._update_spectrograms()
            self._update_biometrics()
            if self.mode == "denoise":
                st = self.dn.stats
                if st.blocks % 10 == 0:
                    log(f"level in={st.in_dbfs:.1f} out={st.out_dbfs:.1f} dBFS, "
                        f"blocks={st.blocks}, dropouts={self.overruns}")
                if st.in_dbfs < -90:
                    self._silent_s += UI_FPS_MS / 1000
                    if self._silent_s > 5:
                        self.tip_lbl.configure(
                            text="⚠  No signal from microphone (−90 dBFS). "
                                 "Check your mic is connected and streaming.",
                            text_color=COLOR_OFF)
                else:
                    self._silent_s = 0
                    self.tip_lbl.configure(
                        text="✓  Receiving audio — use headphones to avoid feedback",
                        text_color=COLOR_ON)
        except Exception as e:    # keep the UI alive no matter what
            log(f"UI tick error: {e!r}")
        self.root.after(UI_FPS_MS, self._tick)

    def _update_biometrics(self):
        # Update the Live Biometric Match Meter
        if self.mode == "denoise" and self.dn is not None and (self.isolate_var.get() or self.live_cocktail_var.get()):
            if self.live_cocktail_var.get():
                score = getattr(self.dn, "_cocktail_score", 0.0)
            else:
                score = getattr(self.dn, "_sv_score", 0.0)
            # Clip between 0 and 1
            score = max(0.0, min(1.0, score))
            self.match_bar.set(score)
            if hasattr(self, '_draw_gauge'):
                self._draw_gauge(score)
            pct = int(score * 100)
            
            thresh = self.threshold_var.get()
            if score >= thresh:
                self.match_bar.configure(progress_color="#00ff88") # Green/Match
                self.match_val_lbl.configure(text=f"{pct}% (MATCH)", text_color="#00ff88")
            else:
                self.match_bar.configure(progress_color="#ff4444") # Red/Stranger
                self.match_val_lbl.configure(text=f"{pct}% (MUTED)", text_color="#ff4444")
        elif hasattr(self, "match_bar"):
            self.match_bar.set(0)
            if hasattr(self, '_draw_gauge'):
                self._draw_gauge(0.0)
            self.match_val_lbl.configure(text="0%", text_color=COLOR_DIM)
            self.match_bar.configure(progress_color=COLOR_ACCENT)

    def _update_stats(self):
        if self.mode != "denoise" or self.dn.stats.blocks == 0:
            return
        st = self.dn.stats
        g = self.stats_lbls
        g["snr"].configure(text=f"{st.snr_est_db:5.1f} dB")
        g["supp"].configure(text=f"{st.suppression_db:5.1f} dB")
        g["in"].configure(text=f"{st.in_dbfs:6.1f} dBFS")
        g["out"].configure(text=f"{st.out_dbfs:6.1f} dBFS")
        g["spch"].configure(text=f"{st.speech_presence * 100:5.1f} %")
        total_lat = self.block_ms if self.mode == "denoise" else int(self.block_sel.get())
        g["lat"].configure(
            text=f"{total_lat + 20:.0f} ms" if self.mode == "denoise" else "—")
        g["infer"].configure(text=f"{st.infer_ms:5.1f} ms")
        g["rt"].configure(text=f"{st.rt_factor:5.2f} x")
        g["blocks"].configure(text=f"{st.blocks}")
        g["over"].configure(text=f"{self.overruns}")
        g["model"].configure(text="DeepFilterNet3")
        g["params"].configure(text=f"{self.dn.n_params:,}")
        g["bins"].configure(text=f"{self.dn.nb_erb} / {self.dn.nb_df}")
        g["fft"].configure(text=f"{self.dn.fft_size}/{self.dn.hop}/{self.dn.sr // 1000}k")
        g["srio"].configure(
            text=f"{self.dn.sr} Hz" + (f" • {self.block_ms}ms" if self.running else ""))
        if self.running and self.start_time:
            up = time.perf_counter() - self.start_time
            g["uptime"].configure(text=f"{int(up // 60):02d}:{int(up % 60):02d}")
        else:
            g["uptime"].configure(text="—")

    def _render_hist(self, hist, is_enh) -> Image.Image:
        if hist is None or hist.shape[1] < 2:
            return Image.new("RGB", (IMG_W, IMG_H), "black")
        arr = hist[:, -HIST_FRAMES:]
        
        # 1. CROP & FLIP (High Graph effect): Human speech lives in the lower 150 frequency bins.
        # By discarding the top 300 empty bins, the voice stretches vertically to fill the ENTIRE graph!
        # We also flip it so the deep bass is at the bottom, creating a massive visual "spike" effect.
        arr = arr[:80, :]
        arr = np.flipud(arr)
        
        # 2. PUNCHY AUTO-GAINER: Optimized for maximum visual presentation impact
        ceiling = float(np.percentile(arr, 99.8))
        if ceiling < -90.0:
            ceiling = -30.0  
        vmax_attr = "_spec_vmax_enh" if is_enh else "_spec_vmax_noisy"
        if not hasattr(self, vmax_attr):
            setattr(self, vmax_attr, ceiling)
        
        # Medium speed tracking
        curr_vmax = getattr(self, vmax_attr)
        new_vmax = 0.95 * curr_vmax + 0.05 * max(ceiling, -50.0)
        setattr(self, vmax_attr, new_vmax)
        
        # 3. HIGH CONTRAST COLORS: Tight 45dB range. 
        # We intentionally offset the math so the voice clips into the brightest yellow and orange colors!
        norm = np.clip((arr - (new_vmax - 50)) / 45.0, 0, 1)
        rgb = MAGMA[(norm * 255).astype(np.uint8)]    # [rows, T, 3]
        return Image.fromarray(rgb).resize((IMG_W, IMG_H), Image.BILINEAR)

    def _update_spectrograms(self):
        from PIL import ImageDraw
        
        is_bypass = (getattr(self, "mode", None) == "bypass")
        
        for lbl, is_enh in [
            (self.spec_noisy, False),
            (self.spec_enh, True),
        ]:
            if is_bypass and is_enh:
                # In bypass mode, the AI is completely turned off to save CPU.
                # Only blank the Enhanced output graph, leave the Microphone graph running.
                img = Image.new("RGB", (IMG_W, IMG_H), "black")
                draw = ImageDraw.Draw(img)
                text = "AI OFF (BYPASS MODE)"
                # Just center it roughly
                draw.text((IMG_W//2 - 60, IMG_H//2 - 5), text, fill="yellow")
            else:
                hist = self.dn.enh_hist if is_enh else self.dn.spec_hist
                if hist is None:
                    continue
                img = self._render_hist(hist, is_enh)
                
            photo = ImageTk.PhotoImage(img)
            lbl.configure(image=photo, width=IMG_W, height=IMG_H)
            lbl.image = photo

    # ═════════════════════════════════════════════════════════════════════
    #  EXIT
    # ═════════════════════════════════════════════════════════════════════

    def _exit(self):
        try:
            self._stop_all()
        finally:
            self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = ctk.CTk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = AudioDenoiserApp(root)
    root.protocol("WM_DELETE_WINDOW", app._exit)
    root.mainloop()


if __name__ == "__main__":
    main()
