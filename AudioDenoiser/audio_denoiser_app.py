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
import threading
import time
import traceback
import tkinter as tk
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
        root.title("AudioDenoiser")
        root.geometry("960x1060")
        root.minsize(900, 800)

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
        """Top-level layout: header + tabview."""
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=(12, 16))

        self._build_header(main)

        self.tabview = ctk.CTkTabview(main, corner_radius=12)
        self.tabview.pack(fill="both", expand=True, pady=(10, 0))
        self.tabview.add("  🎤 Live  ")

        self._build_live_tab()

    def _build_header(self, parent):
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(fill="x")

        # Left side: Title
        left = ctk.CTkFrame(hdr, fg_color="transparent")
        left.pack(side="left")
        ctk.CTkLabel(left, text="AudioDenoiser",
                     font=ctk.CTkFont(size=26, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(left, text="Real-time noise suppression  •  DeepFilterNet3",
                     font=ctk.CTkFont(size=12), text_color=COLOR_DIM).pack(anchor="w")

        # Center: Engine controls
        mid = ctk.CTkFrame(hdr, fg_color="transparent")
        mid.pack(side="left", padx=(40, 0))

        self.onoff_btn = ctk.CTkButton(
            mid, text="●  OFF", width=130, height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_OFF, hover_color=COLOR_OFF_HOVER,
            corner_radius=8, command=self._toggle_main)
        self.onoff_btn.pack(side="left", padx=(0, 8))

        self.repeat_btn = ctk.CTkButton(
            mid, text="🔊  Repeat: ON", width=160, height=38,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLOR_ON, hover_color=COLOR_ON_HOVER,
            corner_radius=8, command=self._toggle_repeat)
        self.repeat_btn.pack(side="left", padx=(0, 16))

        self.rec_btn = ctk.CTkButton(
            mid, text="🔴 Record", width=110, height=32,
            fg_color="gray25", hover_color="gray35",
            command=self._toggle_record)
        self.rec_btn.pack(side="left", padx=(0, 16))

        self.tip_lbl = ctk.CTkLabel(
            mid, text="", font=ctk.CTkFont(size=11),
            text_color=COLOR_DIM, wraplength=400, anchor="w", justify="left")
        self.tip_lbl.pack(side="left", fill="x", expand=True)

    # ── helpers ──────────────────────────────────────────────────────────

    def _card(self, parent, title: str):
        """Create a titled card frame.  Returns the inner content frame."""
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.pack(fill="x", pady=(0, 10), padx=2)
        if title:
            ctk.CTkLabel(card, text=title,
                         font=ctk.CTkFont(size=14, weight="bold")
                         ).pack(anchor="w", padx=16, pady=(12, 0))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=(8, 14))
        return inner

    def _unbind_slider_scroll(self, slider):
        """Prevent CTkSlider from capturing mouse wheel so the page scrolls."""
        # CTkSlider changes value on scroll, and CTkScrollableFrame explicitly ignores
        # scroll events originating from sliders. We need to overwrite the slider's scroll
        # value change AND explicitly forward the scroll event to the parent canvas.
        if hasattr(slider, "_canvas"):
            def forward_scroll(e):
                # find the nearest scrollable frame parent canvas
                w = slider.master
                while w:
                    if isinstance(w, ctk.CTkScrollableFrame):
                        if sys.platform.startswith("win"):
                            w._parent_canvas.yview("scroll", -int(e.delta), "units")
                        return
                    w = w.master
            # NO add="+" here. We WANT to overwrite the default slider scroll behavior.
            slider._canvas.bind("<MouseWheel>", forward_scroll)


    def _speed_up_scroll(self, scroll_frame):
        """Increase the hardcoded scroll sensitivity of CTkScrollableFrame."""
        original_handler = scroll_frame._mouse_wheel_all
        def fast_scroll(event):
            # Normal delta on Windows is 120. CTk divides by 6 (20 units).
            # We use the raw delta (120 units) to make it 6x faster.
            if sys.platform.startswith("win") and scroll_frame._check_if_valid_scroll(event.widget):
                if not scroll_frame._shift_pressed and scroll_frame._parent_canvas.yview() != (0.0, 1.0):
                    scroll_frame._parent_canvas.yview("scroll", -int(event.delta), "units")
                    return
            original_handler(event)
        scroll_frame._mouse_wheel_all = fast_scroll

    # ── Live tab ─────────────────────────────────────────────────────────

    def _build_live_tab(self):
        tab = self.tabview.tab("  🎤 Live  ")
        
        # Scrollable area for everything
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        self._speed_up_scroll(scroll)

        self._build_spectrograms_card(scroll)
        self._build_network_card(scroll)
        self._build_device_card(scroll)
        self._build_voice_card(scroll)
        self._build_controls_card(scroll, "live")
        self._build_stats_card(scroll)

    def _build_network_card(self, parent):
        c = self._card(parent, "Remote Control Network (MQTT)")
        
        net_frm = ctk.CTkFrame(c, fg_color="transparent")
        net_frm.pack(fill="x")
        
        ctk.CTkLabel(net_frm, text="Device ID:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(0,10))
        
        self.net_id_var = tk.StringVar(value="pc1")
        self.net_id_entry = ctk.CTkEntry(net_frm, textvariable=self.net_id_var, width=120)
        self.net_id_entry.pack(side="left", padx=(0,20))
        
        self.net_connect_btn = ctk.CTkButton(net_frm, text="Connect to Cloud", width=120, command=self._toggle_mqtt)
        self.net_connect_btn.pack(side="left")
        
        self.net_status_lbl = ctk.CTkLabel(net_frm, text="Disconnected", font=ctk.CTkFont(size=11), text_color=COLOR_DIM)
        self.net_status_lbl.pack(side="left", padx=(20,0))
        
        self.admin_override_lbl = ctk.CTkLabel(c, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00ff88")
        self.admin_override_lbl.pack(anchor="w", pady=(5,0))

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


    def _build_device_card(self, parent):
        c = self._card(parent, "🎙️  Device Settings")

        # ── device selection ──────────────────────────────────
        dev = ctk.CTkFrame(c, fg_color="transparent")
        dev.pack(fill="x")
        dev.columnconfigure(1, weight=1)

        labels = ["Microphone", "Output", "Block Size"]
        for i, txt in enumerate(labels):
            ctk.CTkLabel(dev, text=txt, font=ctk.CTkFont(size=12)).grid(
                row=i, column=0, sticky="w", padx=(0, 12), pady=3)

        self.in_dev = ctk.CTkComboBox(dev, state="readonly", width=420,
                                       font=ctk.CTkFont(size=11))
        self.in_dev.grid(row=0, column=1, sticky="ew", pady=3)

        self.out_dev = ctk.CTkComboBox(dev, state="readonly", width=420,
                                        font=ctk.CTkFont(size=11))
        self.out_dev.grid(row=1, column=1, sticky="ew", pady=3)

        block_row = ctk.CTkFrame(dev, fg_color="transparent")
        block_row.grid(row=2, column=1, sticky="w", pady=3)
        self.block_sel = ctk.CTkComboBox(block_row, state="readonly", width=90,
                                          values=["40", "100", "200", "400"],
                                          font=ctk.CTkFont(size=11))
        self.block_sel.set("100")
        self.block_sel.pack(side="left")
        ctk.CTkLabel(block_row, text="ms  (lower = less latency, higher = better quality)",
                     font=ctk.CTkFont(size=10), text_color=COLOR_DIM).pack(side="left", padx=8)

    def _build_voice_card(self, parent):
        c = self._card(parent, "🎤  Speaker Isolation (Cancel other voices)")
        
        info_frm = ctk.CTkFrame(c, fg_color="transparent")
        info_frm.pack(fill="x", pady=(0, 10))
        
        self.match_meter_frm = ctk.CTkFrame(c, fg_color="transparent")
        self.match_meter_frm.pack(fill="x", pady=(0, 10))
        
        self.match_lbl = ctk.CTkLabel(self.match_meter_frm, text="Biometric Match:", font=ctk.CTkFont(size=11, weight="bold"))
        self.match_lbl.pack(side="left", padx=(0, 10))
        
        self.match_bar = ctk.CTkProgressBar(self.match_meter_frm, width=200, height=12)
        self.match_bar.pack(side="left")
        self.match_bar.set(0)
        
        self.match_val_lbl = ctk.CTkLabel(self.match_meter_frm, text="0%", font=ctk.CTkFont(size=11), text_color=COLOR_DIM)
        self.match_val_lbl.pack(side="left", padx=(10, 0))
        
        profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")
        has_profile = os.path.exists(profile_path)
        
        self.isolate_var = tk.BooleanVar(value=False)
        self.isolate_switch = ctk.CTkSwitch(
            info_frm, text="Isolate My Voice (Fast)",
            variable=self.isolate_var, command=self._toggle_isolate,
            font=ctk.CTkFont(size=12, weight="bold"),
            state="normal" if has_profile else "disabled"
        )
        self.isolate_switch.pack(side="left", padx=(0, 25))
        
        self.live_cocktail_var = tk.BooleanVar(value=False)
        self.live_cocktail_switch = ctk.CTkSwitch(
            info_frm, text="Cocktail Party Separation (3s Delay)",
            variable=self.live_cocktail_var, command=self._toggle_cocktail,
            font=ctk.CTkFont(size=12, weight="bold"),
            state="normal" if has_profile else "disabled"
        )
        self.live_cocktail_switch.pack(side="left", padx=(0, 25))
        
        self.enroll_btn = ctk.CTkButton(
            info_frm, text="🎙️ Re-enroll Voice" if has_profile else "🎙️ Enroll My Voice", 
            width=140, height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLOR_ACCENT, hover_color=COLOR_ON,
            command=self._enroll_voice)
        self.enroll_btn.pack(side="left", padx=(0, 10))

        self.play_voice_btn = ctk.CTkButton(
            info_frm, text="▶️ Listen", width=80, height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="gray25", hover_color="gray35",
            state="normal" if has_profile else "disabled",
            command=self._play_voice)
        self.play_voice_btn.pack(side="left", padx=(0, 15))
        
        self.enroll_prompt = ctk.CTkLabel(
            info_frm, 
            text="Record a 15-second voice sample in a quiet room to generate your Biometric Centroid Vector.",
            font=ctk.CTkFont(size=11), text_color=COLOR_DIM, justify="left", wraplength=350)
        self.enroll_prompt.pack(side="left")

    def _toggle_isolate(self):
        if self.dn is not None:
            self.dn.isolate_speaker = self.isolate_var.get()
            if self.isolate_var.get():
                self.dn.reload_profile()
        self._broadcast_state()

    def _toggle_cocktail(self):
        if self.dn is not None:
            enabled = self.live_cocktail_var.get()
            if enabled:
                self.live_cocktail_switch.configure(text="Loading Cocktail AI...", state="disabled")
                self.tip_lbl.configure(text="⏳ Loading massive SepFormer AI into memory (takes a few secs)...", text_color="yellow")
                self.root.update_idletasks()
                
                def load_task():
                    try:
                        self.dn.set_cocktail_mode(True)
                        self.root.after(0, lambda: self.live_cocktail_switch.configure(text="Cocktail Party Separation (3s Delay)", state="normal"))
                        self.root.after(0, lambda: self.tip_lbl.configure(text="✅ Cocktail AI Active! (Expect 3s audio delay)", text_color=COLOR_ON))
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
            self.enroll_prompt.configure(text="⚠️ Please select an output device first!")
            return
            
        self.play_voice_btn.configure(state="disabled")
        self.enroll_prompt.configure(text="▶️ Playing your saved Voice Print...", text_color=COLOR_ON)
        
        def play_task():
            try:
                import soundfile as sf
                data, fs = sf.read(profile_path)
                sd.play(data, fs, device=dev_out)
                sd.wait()
                self.enroll_prompt.configure(text="✅ Finished playing back your Voice Print.", text_color=COLOR_DIM)
            except Exception as e:
                self.enroll_prompt.configure(text=f"❌ Error playing: {e}", text_color=COLOR_OFF)
            finally:
                self.play_voice_btn.configure(state="normal")
                
        threading.Thread(target=play_task, daemon=True).start()

    def _enroll_voice(self):
        try:
            dev_in = int(self.in_dev.get().split(":")[0])
        except (ValueError, IndexError):
            self.enroll_prompt.configure(text="❌ Please select a microphone first!")
            return

        self.enroll_btn.configure(state="disabled")
        self.play_voice_btn.configure(state="disabled")
        self.enroll_prompt.configure(text="🔴 RECORDING (20s)...", text_color="orange")
        
        abort_flag = [False]
        
        # --- Teleprompter Overlay ---
        prompt_win = ctk.CTkToplevel(self.root)
        
        def on_close_prompt():
            abort_flag[0] = True
            import sounddevice as sd
            sd.stop()
            prompt_win.destroy()
            self.enroll_prompt.configure(text="❌ Enrollment aborted by user.", text_color="#FF4444")
            self.enroll_btn.configure(state="normal", text="🔄 Re-enroll Voice")
            if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")):
                self.play_voice_btn.configure(state="normal")
                self.isolate_switch.configure(state="normal")

        prompt_win.protocol("WM_DELETE_WINDOW", on_close_prompt)
        prompt_win.title("Voice Enrollment")
        prompt_win.geometry("500x350")
        prompt_win.attributes("-topmost", True)
        prompt_win.geometry(f"+{self.root.winfo_x() + 100}+{self.root.winfo_y() + 50}")
        
        lbl_title = ctk.CTkLabel(prompt_win, text="🔴 RECORDING IN PROGRESS", text_color="#FF4444", font=ctk.CTkFont(size=20, weight="bold"))
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
                
                self.enroll_prompt.configure(text="🧹 Purifying voice print using DeepFilterNet...", text_color="yellow")
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
                
                self.enroll_prompt.configure(text="✅ Voice Print saved! The AI will now use this to filter out other voices.", text_color=COLOR_ON)
                self.enroll_btn.configure(text="🔄 Re-enroll Voice")
            except Exception as e:
                self.enroll_prompt.configure(text=f"❌ Error recording: {e}", text_color=COLOR_OFF)
            finally:
                self.enroll_btn.configure(state="normal")
                if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")):
                    self.play_voice_btn.configure(state="normal")
                    self.isolate_switch.configure(state="normal")
                
                
        threading.Thread(target=record_task, daemon=True).start()

    def _build_controls_card(self, parent, tab: str):
        """Audio controls: attenuation (live only) + mix/gain/hpf/floor + post-filter."""
        title = "🎛️  Audio Controls" if tab == "live" else "🎛️  Audio Controls  (Files)"
        c = self._card(parent, title)
        c.columnconfigure(1, weight=1)
        setattr(self, f"fx_lbls_{tab}", {})

        cur_row = 0

        if tab == "live":
            # ── Environment Profile ──────────────────────────────────────
            profile_fr = ctk.CTkFrame(c, fg_color="transparent")
            profile_fr.grid(row=0, column=0, columnspan=3, sticky="ew", padx=16, pady=(0, 10))
            
            ctk.CTkLabel(profile_fr, text="🌍 Environment Profile:", 
                         font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(0, 12))
            self.profile_var = tk.StringVar(value="Custom")
            self.profile_sel = ctk.CTkComboBox(
                profile_fr, variable=self.profile_var, 
                values=["Custom", "Traffic (Max Suppression)", "Classroom (Babble Control)", "Home (Natural Mix)"],
                command=self._on_profile_change, width=250, state="readonly"
            )
            self.profile_sel.pack(side="left")
            
            sep1 = ctk.CTkFrame(c, height=1, fg_color="gray30")
            sep1.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 6))

            # ── noise attenuation ────────────────────────────────────────
            ctk.CTkLabel(c, text="Noise Attenuation",
                         font=ctk.CTkFont(size=13, weight="bold")).grid(
                row=2, column=0, sticky="w", padx=(0, 16), pady=6)
            self.atten_var = tk.DoubleVar(value=self.params["live"]["atten"])
            atten_s = ctk.CTkSlider(c, from_=0, to=100, variable=self.atten_var,
                          command=lambda v: self._on_param("live", "atten", v),
                          height=20)
            atten_s.grid(row=2, column=1, sticky="ew", padx=4, pady=6)
            self._unbind_slider_scroll(atten_s)
            self.atten_lbl = ctk.CTkLabel(
                c, text=f'{int(self.params["live"]["atten"])} dB',
                font=ctk.CTkFont(size=13, weight="bold"), width=80, anchor="e")
            self.atten_lbl.grid(row=2, column=2, padx=(8, 0), pady=6)

            # visual separator
            sep2 = ctk.CTkFrame(c, height=1, fg_color="gray30")
            sep2.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
            cur_row = 4

        # ── per-param sliders ────────────────────────────────────────
        def _slider(row, name, label, frm, to, init, fmt):
            ctk.CTkLabel(c, text=label, font=ctk.CTkFont(size=12)).grid(
                row=row, column=0, sticky="w", padx=(0, 16), pady=5)
            var = tk.DoubleVar(value=init)
            s = ctk.CTkSlider(c, from_=frm, to=to, variable=var,
                          command=lambda v, n=name, t=tab: self._on_param(t, n, v))
            s.grid(row=row, column=1, sticky="ew", padx=4, pady=5)
            self._unbind_slider_scroll(s)
            lbl = ctk.CTkLabel(c, text=fmt(init),
                               font=ctk.CTkFont(size=12, weight="bold"),
                               width=80, anchor="e")
            lbl.grid(row=row, column=2, padx=(8, 0), pady=5)
            getattr(self, f"fx_lbls_{tab}")[name] = (var, lbl, fmt)

        p = self.params[tab]
        _slider(cur_row + 0, "mix",   "Dry / Wet Mix",       0, 100, p["mix"],
                lambda v: f"{round(v)} %")
        _slider(cur_row + 1, "gain",  "Output Gain",        -12,  12, p["gain"],
                lambda v: f"{round(v):+d} dB")
        _slider(cur_row + 2, "hpf",   "High-Pass Filter",    0, 400, p["hpf"],
                lambda v: "off" if round(v / 20) * 20 < 20
                          else f"{round(v / 20) * 20} Hz")
        _slider(cur_row + 3, "floor", "Max Suppression",      0, 100, p["floor"],
                lambda v: "off" if round(v) >= 100 else f"{round(v)} dB")

        # ── post-filter checkbox ─────────────────────────────────────
        pf_row = cur_row + 4
        siren_row = cur_row + 5
        if tab == "live":
            self.pf_var = tk.BooleanVar(value=self.pf)
            self.pf_cb = ctk.CTkCheckBox(
                c, text="Aggressive post-filter  (reloads model)",
                variable=self.pf_var, command=self._on_pf,
                font=ctk.CTkFont(size=11)
            )
            self.pf_cb.grid(row=pf_row, column=0, columnspan=3, sticky="w", pady=(10, 0))

            self.siren_var = tk.BooleanVar(value=False)
            self.siren_cb = ctk.CTkCheckBox(
                c, text="🚨 Siren & Whistle Killer  (Spectral Median Filter)",
                variable=self.siren_var, command=self._on_siren,
                font=ctk.CTkFont(size=11)
            )
            self.siren_cb.grid(row=siren_row, column=0, columnspan=3, sticky="w", pady=(6, 0))
        else:
            self.pf_var_file = self.pf_var   # shared state, one model
            ctk.CTkLabel(
                c, text="Post-filter and model options are set on the Live tab.",
                font=ctk.CTkFont(size=11), text_color=COLOR_DIM
            ).grid(row=pf_row, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def _build_stats_card(self, parent):
        c = self._card(parent, "📊  Live Monitor")
        self.stats_lbls = {}

        # ── primary stats: 2 × 4 mini-cards ──────────────────────────
        grid = ctk.CTkFrame(c, fg_color="transparent")
        grid.pack(fill="x", pady=(0, 8))
        for col in range(4):
            grid.columnconfigure(col, weight=1)

        primary = [
            ("Estimated SNR", "snr"),   ("Attenuation",     "supp"),
            ("Input Level",   "in"),    ("Output Level",    "out"),
            ("Speech Presence","spch"), ("Total Latency",   "lat"),
            ("Inference Time","infer"), ("Realtime Factor", "rt"),
        ]
        for i, (name, key) in enumerate(primary):
            r, col = divmod(i, 4)
            box = ctk.CTkFrame(grid, corner_radius=8, fg_color=COLOR_CARD_INNER)
            box.grid(row=r, column=col, padx=3, pady=3, sticky="nsew")
            ctk.CTkLabel(box, text=name, font=ctk.CTkFont(size=9),
                         text_color=COLOR_DIM).pack(anchor="w", padx=10, pady=(7, 0))
            lbl = ctk.CTkLabel(box, text="—",
                               font=ctk.CTkFont(size=14, weight="bold"))
            lbl.pack(anchor="w", padx=10, pady=(0, 7))
            self.stats_lbls[key] = lbl

        # ── secondary stats: compact rows ────────────────────────────
        info = ctk.CTkFrame(c, fg_color="transparent")
        info.pack(fill="x")
        for col in range(4):
            info.columnconfigure(col, weight=1)

        secondary = [
            ("Blocks",     "blocks"), ("Dropouts",  "over"),
            ("Uptime",     "uptime"), ("Model",     "model"),
            ("Parameters", "params"), ("ERB / DF",  "bins"),
            ("FFT/Hop/SR", "fft"),    ("Rate",      "srio"),
        ]
        for i, (name, key) in enumerate(secondary):
            r, col = divmod(i, 4)
            row_fr = ctk.CTkFrame(info, fg_color="transparent")
            row_fr.grid(row=r, column=col, padx=6, pady=2, sticky="w")
            ctk.CTkLabel(row_fr, text=f"{name}:", font=ctk.CTkFont(size=10),
                         text_color=COLOR_DIM).pack(side="left")
            lbl = ctk.CTkLabel(row_fr, text="—",
                               font=ctk.CTkFont(size=10, weight="bold"))
            lbl.pack(side="left", padx=(4, 0))
            self.stats_lbls[key] = lbl

    def _build_spectrograms_card(self, parent):
        c = self._card(parent, "📈  Spectrograms")

        for title, attr in [("Microphone  (Noisy)", "spec_noisy"),
                            ("Enhanced Output",     "spec_enh")]:
            ctk.CTkLabel(c, text=title, font=ctk.CTkFont(size=11),
                         text_color=COLOR_DIM).pack(anchor="w", pady=(4, 2))
            # Use plain tk.Label for performant 10-fps image updates
            ph = tk.PhotoImage(width=IMG_W, height=IMG_H)
            lbl = tk.Label(c, image=ph, bg="black", borderwidth=0,
                           highlightthickness=0)
            lbl.image = ph
            lbl.pack(pady=(0, 6))
            setattr(self, attr, lbl)

    # ── Files tab ────────────────────────────────────────────────────────

    def _build_files_tab(self):
        tab = self.tabview.tab("  📁 Files  ")
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        self._speed_up_scroll(scroll)

        self._build_controls_card(scroll, "file")

        c = self._card(scroll, "📂  Denoise Audio Files")

        # ── file list + buttons ──────────────────────────────────────
        body = ctk.CTkFrame(c, fg_color="transparent")
        body.pack(fill="x")
        body.columnconfigure(0, weight=1)

        # Dark-styled Listbox (no CTk equivalent)
        self.file_list = tk.Listbox(
            body, height=8, selectmode="extended",
            bg="#1a1a2e", fg="#cbd5e1", selectbackground="#1f6aa5",
            selectforeground="white", borderwidth=0,
            highlightthickness=1, highlightcolor="#334155",
            highlightbackground="#252540",
            font=("Segoe UI", 10), activestyle="none")
        self.file_list.grid(row=0, column=0, rowspan=5, sticky="nsew",
                            padx=(0, 12), pady=2)

        btns = [
            ("📄  Add Files…",       self._add_files,    0, False),
            ("✕   Remove Selected",  self._remove_files, 1, False),
            ("🗑️  Clear All",        self._clear_files,  2, False),
        ]
        for text, cmd, row, accent in btns:
            ctk.CTkButton(body, text=text, command=cmd, width=160, height=34,
                          fg_color="gray25", hover_color="gray35",
                          font=ctk.CTkFont(size=11), corner_radius=8
                          ).grid(row=row, column=1, pady=2, sticky="ew")

        # denoise button (accent)
        self.denoise_btn = ctk.CTkButton(
            body, text="▶  Denoise Files", command=self._denoise_files,
            width=160, height=40, corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"))
        self.denoise_btn.grid(row=3, column=1, pady=(12, 2), sticky="ew")

        ctk.CTkButton(body, text="📂  Open Output Folder",
                      command=self._open_out_dir, width=160, height=34,
                      fg_color="gray25", hover_color="gray35",
                      font=ctk.CTkFont(size=11), corner_radius=8
                      ).grid(row=4, column=1, pady=2, sticky="ew")

        # Target Speaker Separation Switch
        profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")
        has_profile = os.path.exists(profile_path)
        self.file_separate_var = tk.BooleanVar(value=False)
        self.file_separate_switch = ctk.CTkSwitch(
            c, text="Cocktail Party Separation (Requires Voice Print)",
            variable=self.file_separate_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            state="normal" if has_profile else "disabled"
        )
        self.file_separate_switch.pack(fill="x", pady=(15, 0), padx=10)

        # ── status label ─────────────────────────────────────────────
        self.file_status = ctk.CTkLabel(
            c,
            text="Supported: wav, mp3, flac, ogg, m4a, aac, wma\n"
                 "Output: 48 kHz mono WAV → AudioDenoiser\\output\\<name>_denoised.wav",
            font=ctk.CTkFont(size=11), text_color=COLOR_DIM,
            justify="left", anchor="w", wraplength=600)
        self.file_status.pack(fill="x", pady=(8, 0))

    # ═════════════════════════════════════════════════════════════════════
    #  DEVICE MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════

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
        if 0 <= din < len(devs) and ins:
            match = [t for t in ins if t.startswith(f"{din}:")]
            self.in_dev.set(match[0] if match else ins[0])
        if 0 <= dout < len(devs) and outs:
            match = [t for t in outs if t.startswith(f"{dout}:")]
            self.out_dev.set(match[0] if match else outs[0])

    # ═════════════════════════════════════════════════════════════════════
    #  PARAMETERS
    # ═════════════════════════════════════════════════════════════════════

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

    def _update_buttons(self):
        denoise_on = self.mode == "denoise"
        self.onoff_btn.configure(
            text="●  ON" if denoise_on else "●  OFF",
            fg_color=COLOR_ON if denoise_on else COLOR_OFF,
            hover_color=COLOR_ON_HOVER if denoise_on else COLOR_OFF_HOVER)
            
        if self.repeat:
            self.repeat_btn.configure(
                text="🔊  Repeat: ON",
                fg_color=COLOR_ON,
                hover_color=COLOR_ON_HOVER)
        else:
            self.repeat_btn.configure(
                text="🔇  Repeat: OFF",
                fg_color=COLOR_OFF, hover_color=COLOR_OFF_HOVER)
                
        # Update tip label to explain bypass mode
        if self.mode == "bypass":
            self.tip_lbl.configure(text="ℹ  Bypass Mode: Playing raw original microphone audio.", text_color="yellow")
        elif self.mode == "denoise":
            self.tip_lbl.configure(text="⚡  Denoising Active", text_color=COLOR_ON)
        else:
            self.tip_lbl.configure(text="")

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
        # Update the Live Biometric Match Meter if Voice Isolation is active
        if self.mode == "denoise" and self.dn is not None and getattr(self, "isolate_var", None) and self.isolate_var.get():
            score = getattr(self.dn, "_latest_score", 0.0)
            # Clip between 0 and 1
            score = max(0.0, min(1.0, score))
            self.match_bar.set(score)
            pct = int(score * 100)
            
            # Color code based on threshold (0.28)
            if score > 0.28:
                self.match_bar.configure(progress_color="#00ff88") # Green/Match
                self.match_val_lbl.configure(text=f"{pct}% (MATCH)", text_color="#00ff88")
            else:
                self.match_bar.configure(progress_color="#ff4444") # Red/Stranger
                self.match_val_lbl.configure(text=f"{pct}% (MUTED)", text_color="#ff4444")
        elif hasattr(self, "match_bar"):
            self.match_bar.set(0)
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

    def _render_hist(self, hist) -> Image.Image:
        if hist is None or hist.shape[1] < 2:
            return Image.new("RGB", (IMG_W, IMG_H), "black")
        arr = hist[:, -HIST_FRAMES:]
        # pool 481 frequency bins -> 160 rows (axis 0!)
        rows = 160
        arr = arr[: rows * 3, :].reshape(rows, 3, -1).mean(axis=1)
        # Slow Auto-Gainer: Adapts to quiet microphones, but shifts VERY slowly 
        # so it doesn't cause distracting color flashes when you speak.
        ceiling = float(np.percentile(arr, 99.8))
        if ceiling < -90.0:
            ceiling = -30.0  
        if not hasattr(self, "_spec_vmax"):
            self._spec_vmax = ceiling
        
        # 0.99 makes it extremely slow and stable (takes several seconds to shift)
        self._spec_vmax = 0.99 * self._spec_vmax + 0.01 * max(ceiling, -50.0)
        
        # Expanded 80dB dynamic range makes the visuals much smoother and easier to read
        norm = np.clip((arr - (self._spec_vmax - 80)) / 80.0, 0, 1)
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
                img = self._render_hist(hist)
                
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
