import os
import re

file_path = r"c:\Projects\AML LAB\AudioDenoiser\AudioDenoiser\audio_denoiser_app.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add colors and math import
content = content.replace("import tkinter as tk", "import tkinter as tk\nimport math")
colors = """
BG = '#0D0D12'
PANEL = '#1A1A24'
PANEL_BORDER = '#2A2A35'
CYAN = '#00E5FF'
GREEN = '#00FF88'
CRIMSON = '#FF3366'
TEXT_PRIMARY = '#E8E8F0'
TEXT_MUTED = '#6B6B80'
TEXT_DIM = '#44445A'
"""
content = content.replace('COLOR_CARD_INNER = ("gray85", "gray17")', 'COLOR_CARD_INNER = ("gray85", "gray17")\n' + colors)

# 2. Modify __init__ to set correct window properties
init_regex = r"(def __init__\(self, root: ctk\.CTk\):.*?)(?=\n    def _build_ui)"
init_match = re.search(init_regex, content, re.DOTALL)
if init_match:
    init_code = init_match.group(1)
    init_code = init_code.replace('root.title("AudioDenoiser")', 'root.title("NeuroAcoustic Engine")')
    init_code = init_code.replace('root.geometry("960x1060")', 'root.geometry("1280x720")\n        root.resizable(False, False)\n        root.configure(fg_color=BG)')
    init_code = init_code.replace('root.minsize(900, 800)', '')
    init_code = init_code.replace('self.remote_controller = None', 'self.remote_controller = None\n        self.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")\n        self._file_busy = False')
    content = content[:init_match.start()] + init_code + content[init_match.end():]

# 3. Replace all UI methods
ui_start = content.find("    def _build_ui(self):")
ui_end = content.find("    # ═════════════════════════════════════════════════════════════════════\n    #  DEVICE MANAGEMENT")

new_ui_methods = """
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

        self.onoff_btn = ctk.CTkButton(top_fr, text="SYSTEM MASTER", width=140, height=36,
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
        
        self.tip_lbl = ctk.CTkLabel(top_fr, text="", font=ctk.CTkFont(size=11), text_color=TEXT_DIM)
        self.tip_lbl.pack(side="left", fill="x", expand=True, padx=10)

        self.stats_lbls = {}
        hdr_stats = ctk.CTkFrame(top_fr, fg_color="transparent")
        hdr_stats.pack(side="right", padx=15, pady=10)
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
        for k in ["blocks", "uptime", "params", "bins", "fft", "srio"]:
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

        ctk.CTkLabel(c, text="🌍 Environment Profile:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w", pady=10)
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
        if not hasattr(self, "atten_lbl"):
            self.atten_lbl = ctk.CTkLabel(c, text=f'{int(self.params["live"]["atten"])} dB', font=ctk.CTkFont(size=12, weight="bold"), width=60)
        self.atten_lbl.grid(row=cur_row, column=2, pady=5)
        cur_row += 1
        
        def _slider(row, name, label, frm, to, init, fmt):
            ctk.CTkLabel(c, text=label, font=ctk.CTkFont(size=11)).grid(row=row, column=0, sticky="w", pady=5)
            if name in self.fx_lbls_live:
                var = self.fx_lbls_live[name][0]
                lbl = self.fx_lbls_live[name][1]
            else:
                var = tk.DoubleVar(value=init)
                lbl = ctk.CTkLabel(c, text=fmt(init), font=ctk.CTkFont(size=11, weight="bold"), width=60)
                self.fx_lbls_live[name] = (var, lbl, fmt)
                self.fx_lbls_file[name] = (var, lbl, fmt)
            s = ctk.CTkSlider(c, from_=frm, to=to, variable=var, command=lambda v, n=name: self._on_param("live", n, v))
            s.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
            lbl = ctk.CTkLabel(c, text=fmt(var.get()), font=ctk.CTkFont(size=11, weight="bold"), width=60)
            lbl.grid(row=row, column=2, pady=5)
            self.fx_lbls_live[name] = (var, lbl, fmt)
            
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
        self.siren_cb = ctk.CTkCheckBox(c, text="🚨 Siren & Whistle Killer", variable=self.siren_var, command=self._on_siren)
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

"""
new_content = content[:ui_start] + new_ui_methods + content[ui_end:]

bio_start = new_content.find("    def _update_biometrics(self):")
bio_end = new_content.find("    def _update_stats(self):")
bio_code = new_content[bio_start:bio_end]
bio_code = bio_code.replace("self.match_bar.set(score)", "self.match_bar.set(score)\n            if hasattr(self, '_draw_gauge'):\n                self._draw_gauge(score)")
bio_code = bio_code.replace("self.match_bar.set(0)", "self.match_bar.set(0)\n            if hasattr(self, '_draw_gauge'):\n                self._draw_gauge(0.0)")
new_content = new_content[:bio_start] + bio_code + new_content[bio_end:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Done.")
