"""AudioDenoiser — real-time microphone noise suppression for Windows.

Two tabs:
  * Live      — mic capture -> DeepFilterNet3 -> speaker output, with
                attenuation, sound customization, spectrograms and live stats.
  * Files     — batch-denoise audio files offline (full-quality path).

Run:  run.bat
Use headphones to avoid feedback from speakers back into the mic.
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageTk

import sounddevice as sd

from denoiser_pipeline import REPO_DIR, StreamingDenoiser

MODEL_DIR = os.path.abspath(os.path.join(REPO_DIR, "models", "DeepFilterNet3"))
HIST_FRAMES = 400  # spectrogram history length in frames (10 ms each)
IMG_W, IMG_H = 800, 145
UI_FPS_MS = 100

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audiodenoiser.log")


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


MAGMA = (colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


class JitterBuffer:
    """Bridges two audio devices running on different clocks (e.g. Audio Relay's
    virtual mic vs. a local sound card). The playback callback pulls a fixed
    number of frames per tick while production drifts against it; without
    compensation you get periodic silence gaps (underrun) or dropped blocks.

    read() continuously micro-resamples (linear interpolation, ±0.4 % max) so
    the consumption rate tracks the fill level around `target` samples. Real
    clock drift is tens of ppm, so the correction is inaudible; the buffer
    depth absorbs network jitter.
    """

    def __init__(self, target_samples: int):
        self.buf = np.zeros(0, dtype=np.float32)
        self.pos = 0.0  # read position (float, fractional-sample accurate)
        self.target = float(target_samples)

    def append(self, x: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        if self.pos > 480000:  # trim consumed head every ~10 s
            cut = int(self.pos)
            self.buf = self.buf[cut:]
            self.pos -= cut

    def read(self, frames: int) -> np.ndarray:
        fill = len(self.buf) - self.pos
        # consumption ratio: pull faster when buffer is full, slower when
        # starved. Correction is tiny (±0.4 %) — clock drift is tens of ppm,
        # so anything larger is audible pitch wobble.
        ratio = float(np.clip(1.0 + (fill - self.target) / self.target * 0.004, 0.996, 1.004))
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

# defaults for each tab's parameter set. floor=24 dB keeps speech audible
# through noise (see README "Tuning"); 100 would mean unlimited suppression.
DEFAULT_PARAMS = {"atten": 100, "mix": 100, "gain": 0, "hpf": 0, "floor": 24}


class AudioDenoiserApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("AudioDenoiser")
        root.configure(bg="#fafafa")
        root.geometry("880x1000")

        self.dn = StreamingDenoiser(MODEL_DIR)
        self.pf = False  # current model post-filter state

        # per-tab parameter values
        self.params = {"live": dict(DEFAULT_PARAMS), "file": dict(DEFAULT_PARAMS)}

        self.running = False
        self.repeat = True  # playback of (filtered or raw) audio through output
        self.mode: str | None = None  # None | 'denoise' | 'bypass'
        self.in_stream = None
        self.out_stream = None
        self.worker = None
        self.in_q: queue.Queue = queue.Queue(maxsize=25)
        self.jb: JitterBuffer | None = None
        self.jb_lock = threading.Lock()
        self.overruns = 0
        self.block_ms = 100
        self.start_time = None
        self._file_busy = False
        self.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

        self._build_ui()
        self._update_buttons()
        self._refresh_devices()
        self.root.after(UI_FPS_MS, self._tick)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=14, pady=(12, 4))
        ttk.Label(header, text="AudioDenoiser", font=("Segoe UI", 20, "bold")).pack(side="left")
        ttk.Button(header, text="exit", command=self._exit).pack(side="right")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=6)
        self.live_tab = ttk.Frame(self.nb)
        self.files_tab = ttk.Frame(self.nb)
        self.nb.add(self.live_tab, text="  Live  ")
        self.nb.add(self.files_tab, text="  Files  ")
        self._build_live_tab()
        self._build_files_tab()

    def _build_live_tab(self):
        tab = self.live_tab
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill="x", padx=4, pady=6)

        self.onoff_btn = tk.Button(
            ctrl, text="OFF", width=10, bg="#d9534f", fg="white",
            font=("Segoe UI", 12, "bold"), relief="flat", command=self._toggle_main,
        )
        self.onoff_btn.pack(side="left", padx=(0, 8))
        self.repeat_btn = tk.Button(
            ctrl, text="Repeat: ON", width=12, bg="#5cb85c", fg="white",
            font=("Segoe UI", 10, "bold"), relief="flat", command=self._toggle_repeat,
        )
        self.repeat_btn.pack(side="left", padx=(0, 16))

        ttk.Label(ctrl, text="Noise Attenuation [dB]").pack(side="left")
        self.atten_var = tk.DoubleVar(value=self.params["live"]["atten"])
        ttk.Scale(ctrl, from_=0, to=100, variable=self.atten_var,
                  command=lambda v: self._on_param("live", "atten", v), length=400).pack(
            side="left", padx=8)
        self.atten_lbl = ttk.Label(ctrl, text=str(self.params["live"]["atten"]), width=5)
        self.atten_lbl.pack(side="left")

        self._build_fx_frame(tab, "live")
        self._build_devices(tab)
        self._build_stats(tab)
        self.spec_noisy = self._spectrogram_panel(tab, "Microphone (Noisy)")
        self.spec_enh = self._spectrogram_panel(tab, "AudioDenoiser Enhanced")

    def _build_fx_frame(self, parent, tab):
        """Sliders: dry/wet, output gain, high-pass + post-filter checkbox."""
        fx = ttk.LabelFrame(parent, text="Sound customization")
        fx.pack(fill="x", padx=4, pady=4)
        setattr(self, f"fx_lbls_{tab}", {})

        def add_slider(col, name, label, frm, to, init, fmt):
            ttk.Label(fx, text=label).grid(row=0, column=col, padx=8, sticky="w")
            var = tk.DoubleVar(value=init)
            ttk.Scale(fx, from_=frm, to=to, variable=var,
                      command=lambda v, n=name, t=tab: self._on_param(t, n, v),
                      length=140).grid(row=1, column=col, padx=8)
            lbl = ttk.Label(fx, text=fmt(init), width=6)
            lbl.grid(row=1, column=col, padx=(156, 0), sticky="w")
            getattr(self, f"fx_lbls_{tab}")[name] = (var, lbl, fmt)

        p = self.params[tab]
        add_slider(0, "mix", "Dry / Wet mix", 0, 100, p["mix"],
                   lambda v: f"{round(v)} %")
        add_slider(1, "gain", "Output gain [dB]", -12, 12, p["gain"],
                   lambda v: f"{round(v):+d} dB")
        add_slider(2, "hpf", "High-pass filter [Hz]", 0, 400, p["hpf"],
                   lambda v: "off" if round(v / 20) * 20 < 20 else f"{round(v / 20) * 20} Hz")
        add_slider(3, "floor", "Max suppression [dB]", 0, 100, p["floor"],
                   lambda v: "off" if round(v) >= 100 else f"{round(v)} dB")
        if tab == "live":
            self.pf_var = tk.BooleanVar(value=self.pf)
            ttk.Checkbutton(
                fx, text="Aggressive post-filter (reloads model)",
                variable=self.pf_var, command=self._on_pf,
            ).grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 4))
        else:
            self.pf_var_file = self.pf_var  # shared state, one model
            ttk.Label(
                fx, text="Post-filter and other model options are set on the Live tab.",
                foreground="#777",
            ).grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 4))
        for c in range(4):
            fx.columnconfigure(c, weight=1)

    def _build_devices(self, tab):
        dev = ttk.Frame(tab)
        dev.pack(fill="x", padx=4, pady=2)
        ttk.Label(dev, text="Microphone:").grid(row=0, column=0, sticky="w")
        self.in_dev = ttk.Combobox(dev, width=48, state="readonly")
        self.in_dev.grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(dev, text="Output:").grid(row=1, column=0, sticky="w")
        self.out_dev = ttk.Combobox(dev, width=48, state="readonly")
        self.out_dev.grid(row=1, column=1, padx=6, pady=2)
        ttk.Label(dev, text="Block size:").grid(row=2, column=0, sticky="w")
        self.block_sel = ttk.Combobox(
            dev, width=8, state="readonly", values=("40", "100", "200", "400")
        )
        self.block_sel.set("100")
        self.block_sel.grid(row=2, column=1, padx=6, pady=2, sticky="w")
        self.tip_lbl = ttk.Label(
            dev, text="Tip: use headphones — loud speakers will echo back into the mic.",
            foreground="#777",
        )
        self.tip_lbl.grid(row=3, column=1, sticky="w", padx=6)

    def _build_stats(self, tab):
        self.stats_lbls = {}
        stats = ttk.LabelFrame(tab, text="Live data")
        stats.pack(fill="x", padx=4, pady=8)
        cols = [
            ("Estimated SNR", "snr"), ("Attenuation applied", "supp"),
            ("Input level", "in"), ("Output level", "out"),
            ("Speech presence", "spch"), ("Total latency", "lat"),
            ("Inference / block", "infer"), ("Realtime factor", "rt"),
            ("Blocks processed", "blocks"), ("Audio dropouts", "over"),
            ("Model", "model"), ("Model parameters", "params"),
            ("ERB bands / DF bins", "bins"), ("FFT / hop / sr", "fft"),
            ("Sample rate in/out", "srio"), ("Uptime", "uptime"),
        ]
        for i, (name, key) in enumerate(cols):
            r, c = divmod(i, 4)
            frame = ttk.Frame(stats)
            frame.grid(row=r, column=c, padx=8, pady=3, sticky="w")
            ttk.Label(frame, text=name, foreground="#666", font=("Segoe UI", 8)).pack(anchor="w")
            lbl = ttk.Label(frame, text="—", font=("Segoe UI", 10, "bold"))
            lbl.pack(anchor="w")
            self.stats_lbls[key] = lbl
        for cidx in range(4):
            stats.columnconfigure(cidx, weight=1)

    def _build_files_tab(self):
        tab = self.files_tab
        self._build_fx_frame(tab, "file")

        files = ttk.LabelFrame(tab, text="Denoise audio files (offline, best quality)")
        files.pack(fill="x", padx=4, pady=8)
        self.file_list = tk.Listbox(files, height=8, selectmode="extended")
        self.file_list.grid(row=0, column=0, rowspan=3, padx=6, pady=4, sticky="we")
        files.columnconfigure(0, weight=1)
        ttk.Button(files, text="Add files…", command=self._add_files).grid(row=0, column=1, padx=4, pady=2, sticky="we")
        ttk.Button(files, text="Remove selected", command=self._remove_files).grid(row=1, column=1, padx=4, pady=2, sticky="we")
        ttk.Button(files, text="Clear", command=self._clear_files).grid(row=2, column=1, padx=4, pady=2, sticky="we")
        self.denoise_btn = ttk.Button(files, text="Denoise files", command=self._denoise_files)
        self.denoise_btn.grid(row=4, column=1, padx=4, pady=(8, 2), sticky="we")
        ttk.Button(files, text="Open output folder", command=self._open_out_dir).grid(row=5, column=1, padx=4, pady=2, sticky="we")
        self.file_status = ttk.Label(
            files,
            text="Supported: wav, mp3, flac, ogg, m4a, aac, wma.\n"
                 "Output: 48 kHz mono WAV in AudioDenoiser\\output\\, named <name>_denoised.wav.\n"
                 "Files use the sliders above (this tab) and the post-filter from the Live tab.",
            foreground="#777", justify="left",
        )
        self.file_status.grid(row=4, column=0, padx=6, sticky="sw")

    def _spectrogram_panel(self, parent, title) -> tk.Label:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=4, pady=2)
        ttk.Label(frame, text=title, font=("Segoe UI", 10)).pack(anchor="w")
        ph = tk.PhotoImage(width=IMG_W, height=IMG_H)  # pixel-sized black placeholder
        lbl = tk.Label(frame, image=ph, bg="black")
        lbl.image = ph
        lbl.pack()
        return lbl

    def _refresh_devices(self):
        devs = sd.query_devices()
        ins, outs = [], []
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0:
                ins.append(f"{i}: {d['name']}")
            if d["max_output_channels"] > 0:
                outs.append(f"{i}: {d['name']}")
        self.in_dev["values"] = ins
        self.out_dev["values"] = outs
        din, dout = sd.default.device
        if 0 <= din < len(ins) and ins:
            match = [t for t in ins if t.startswith(f"{din}:")]
            self.in_dev.set(match[0] if match else ins[0])
        if 0 <= dout < len(outs) and outs:
            match = [t for t in outs if t.startswith(f"{dout}:")]
            self.out_dev.set(match[0] if match else outs[0])

    # ------------------------------------------------------- parameters
    def _on_param(self, tab, name, raw):
        """Slider moved: store value, update label, apply immediately to the
        pipeline (the processing thread picks new values up on the next block;
        the set_* writes are plain attribute assignments, safe to do live)."""
        if name == "atten":
            v = int(round(float(raw)))
            self.params[tab]["atten"] = v
            if tab == "live":
                self.atten_lbl.configure(text=str(v))
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

    def _on_pf(self):
        if self.running:
            self.tip_lbl.configure(
                text="Post-filter change applies the next time you turn OFF→ON.",
                foreground="#777",
            )
            return
        self._ensure_denoiser()

    def _ensure_denoiser(self):
        """Rebuild the denoiser if the post-filter option changed."""
        pf = bool(self.pf_var.get())
        if self.pf == pf:
            return
        self.file_status.configure(text="Reloading model…", foreground="#777")
        self.root.update_idletasks()
        self.dn = StreamingDenoiser(MODEL_DIR, post_filter=pf)
        self.pf = pf
        log(f"model reloaded with post_filter={pf}")
        self.file_status.configure(
            text=f"Model reloaded (post-filter {'ON' if pf else 'OFF'}).", foreground="#777"
        )

    # ------------------------------------------------------------ on/off
    # state matrix (main ON/OFF, Repeat ON/OFF):
    #   OFF/OFF  nothing runs
    #   OFF/ON   raw passthrough: mic -> speaker, unfiltered (bypass mode)
    #   ON /ON   denoised audio played to output
    #   ON /OFF  denoiser runs (stats, spectrograms) but output stays silent

    def _toggle_main(self):
        if self.mode is not None:
            self._stop_all()
        else:
            self._start_engine()

    def _toggle_repeat(self):
        self.repeat = not self.repeat
        self._update_buttons()
        if self.mode is not None:
            # switch live: restart engine in the new mode
            was = self.mode
            self._stop_all()
            if self.repeat:
                self._start_engine(denoise=(was == "denoise"))

    def _update_buttons(self):
        denoise_on = self.mode == "denoise"
        self.onoff_btn.configure(
            text="ON" if denoise_on else "OFF",
            bg="#5cb85c" if denoise_on else "#d9534f",
        )
        if self.repeat:
            self.repeat_btn.configure(
                text="Repeat: ON", bg="#5cb85c" if self.mode is not None else "#777"
            )
        else:
            self.repeat_btn.configure(text="Repeat: OFF", bg="#d9534f")

    def _start_engine(self, denoise: bool = True):
        self._ensure_denoiser()
        self.block_ms = int(self.block_sel.get())
        self.block_n = self.dn.hop * self.block_ms // 10
        if denoise:
            self.dn.reset()
            self._apply_params("live")
        self.overruns = 0
        self.start_time = time.perf_counter()
        dev_in = int(self.in_dev.get().split(":")[0])
        dev_out = int(self.out_dev.get().split(":")[0])

        def in_cb(indata, frames, t, status):
            if status:
                pass
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
                blocksize=self.block_n, device=dev_in, callback=in_cb,
            )
            if self.repeat:
                self.out_stream = sd.OutputStream(
                    samplerate=self.dn.sr, channels=1, dtype="float32",
                    blocksize=self.block_n, device=dev_out, callback=out_cb,
                )
        except Exception as e:
            self.tip_lbl.configure(
                text=f"Could not open audio device: {e}", foreground="#c00"
            )
            log(f"ERROR opening streams: {e}")
            return
        log(f"stream opened: mode={'denoise' if denoise else 'bypass'} "
            f"repeat={self.repeat} in={self.in_dev.get()} out={self.out_dev.get()} "
            f"block={self.block_ms}ms sr={self.dn.sr}")
        self._silent_s = 0
        # ~1.5 blocks of backlog: absorbs clock drift + Audio Relay network
        # jitter with ~150 ms extra delay (block+algo+cushion ≈ 280 ms total)
        self.jb = JitterBuffer(target_samples=int(self.block_n * 1.5))
        self.in_stream.start()
        if self.out_stream:
            self.out_stream.start()
        self.mode = "denoise" if denoise else "bypass"
        self.running = True  # must be set before the worker starts its loop
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
        while self.running:
            try:
                blk = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.mode == "bypass":
                with self.jb_lock:
                    self.jb.append(blk)
                continue
            try:
                out = self.dn.process_block(blk)
            except Exception:
                import traceback

                log("worker crash:\n" + traceback.format_exc())
                self.running = False
                return
            if self.repeat:
                with self.jb_lock:
                    self.jb.append(out)

    # ---------------------------------------------------------- file mode
    def _add_files(self):
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(
            title="Choose audio files to denoise",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma"),
                ("All files", "*.*"),
            ],
        )
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
                text="Turn the live denoiser OFF before processing files.", foreground="#c00"
            )
            return
        files = list(self.file_list.get(0, "end"))
        if not files:
            self.file_status.configure(text="Add some files first.", foreground="#c00")
            return
        self._ensure_denoiser()
        self._file_busy = True
        self.denoise_btn.configure(state="disabled")
        os.makedirs(self.out_dir, exist_ok=True)

        def worker():
            results = []
            for i, src in enumerate(files):
                self._file_progress = (
                    f"Processing {i + 1}/{len(files)}: {os.path.basename(src)}"
                )
                try:
                    stem = os.path.splitext(os.path.basename(src))[0]
                    dst = os.path.join(self.out_dir, f"{stem}_denoised.wav")
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
        self.file_status.configure(text=self._file_progress, foreground="#777")
        if not self._file_done:
            self.root.after(200, self._poll_file_worker)
            return
        self._file_busy = False
        self.denoise_btn.configure(state="normal")
        ok = [r for s, r in self._file_results if s == "ok"]
        fail = [r for s, r in self._file_results if s == "fail"]
        msg = f"Done: {len(ok)} file(s) denoised → {self.out_dir}"
        if fail:
            msg += f"  ({len(fail)} failed — see audiodenoiser.log)"
        self.file_status.configure(text=msg, foreground="#c00" if fail else "#2e7d32")
        log(msg)

    # ------------------------------------------------------------- ticks
    def _tick(self):
        try:
            self._update_stats()
            self._update_spectrograms()
            if self.mode == "denoise":
                st = self.dn.stats
                if st.blocks % 10 == 0:
                    log(f"level in={st.in_dbfs:.1f} out={st.out_dbfs:.1f} dBFS, "
                        f"blocks={st.blocks}, dropouts={self.overruns}")
                if st.in_dbfs < -90:
                    self._silent_s += UI_FPS_MS / 1000
                    if self._silent_s > 5:
                        self.tip_lbl.configure(
                            text="No signal from the microphone (-90 dBFS). "
                                 "If using Audio Relay, make sure the phone is connected "
                                 "and streaming; otherwise pick your real mic above.",
                            foreground="#c00",
                        )
                else:
                    self._silent_s = 0
                    self.tip_lbl.configure(
                        text="Tip: use headphones — loud speakers will echo back into the mic.",
                        foreground="#777",
                    )
        except Exception as e:  # keep the UI alive no matter what
            log(f"UI tick error: {e!r}")
        self.root.after(UI_FPS_MS, self._tick)

    def _update_stats(self):
        st = self.dn.stats
        g = self.stats_lbls
        g["snr"].configure(text=f"{st.snr_est_db:5.1f} dB")
        g["supp"].configure(text=f"{st.suppression_db:5.1f} dB")
        g["in"].configure(text=f"{st.in_dbfs:6.1f} dBFS")
        g["out"].configure(text=f"{st.out_dbfs:6.1f} dBFS")
        g["spch"].configure(text=f"{st.speech_presence * 100:5.1f} %")
        total_lat = self.block_ms if self.mode == "denoise" else int(self.block_sel.get())
        g["lat"].configure(
            text=f"{total_lat + 20:.0f} ms" if self.mode == "denoise" else "—"
        )
        g["infer"].configure(text=f"{st.infer_ms:5.1f} ms")
        g["rt"].configure(text=f"{st.rt_factor:5.2f} x")
        g["blocks"].configure(text=f"{st.blocks}")
        g["over"].configure(text=f"{self.overruns}")
        g["model"].configure(text="DeepFilterNet3")
        g["params"].configure(text=f"{self.dn.n_params:,}")
        g["bins"].configure(text=f"{self.dn.nb_erb} / {self.dn.nb_df}")
        g["fft"].configure(text=f"{self.dn.fft_size} / {self.dn.hop} / {self.dn.sr // 1000} kHz")
        g["srio"].configure(
            text=f"{self.dn.sr} Hz" + (f"  block {self.block_ms} ms" if self.running else "")
        )
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
        # auto-gain: track a slowly moving ceiling so quiet mics stay visible
        ceiling = float(np.percentile(arr, 99.8))
        if not hasattr(self, "_spec_vmax"):
            self._spec_vmax = ceiling
        self._spec_vmax = 0.9 * self._spec_vmax + 0.1 * max(ceiling, -80.0)
        norm = np.clip((arr - (self._spec_vmax - 60)) / 60.0, 0, 1)
        rgb = MAGMA[(norm * 255).astype(np.uint8)]  # [rows, T, 3]
        return Image.fromarray(rgb).resize((IMG_W, IMG_H), Image.BILINEAR)

    def _update_spectrograms(self):
        for lbl, hist in (
            (self.spec_noisy, self.dn.spec_hist),
            (self.spec_enh, self.dn.enh_hist),
        ):
            if hist is None:
                continue
            img = self._render_hist(hist)
            photo = ImageTk.PhotoImage(img)
            lbl.configure(image=photo, width=IMG_W, height=IMG_H)
            lbl.image = photo

    def _exit(self):
        try:
            self._stop_all()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
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
