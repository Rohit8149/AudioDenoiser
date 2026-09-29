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

