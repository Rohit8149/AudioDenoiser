    def _build_controls_card(self, parent, tab: str):
        """Audio controls: attenuation (live only) + mix/gain/hpf/floor + post-filter."""
        title = "≡ƒÄ¢∩╕Å  Audio Controls" if tab == "live" else "≡ƒÄ¢∩╕Å  Audio Controls  (Files)"
        c = self._card(parent, title)
        c.columnconfigure(1, weight=1)
        setattr(self, f"fx_lbls_{tab}", {})

        cur_row = 0

        if tab == "live":
            # ΓöÇΓöÇ Environment Profile ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            profile_fr = ctk.CTkFrame(c, fg_color="transparent")
            profile_fr.grid(row=0, column=0, columnspan=3, sticky="ew", padx=16, pady=(0, 10))
            
            ctk.CTkLabel(profile_fr, text="≡ƒîì Environment Profile:", 
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

            # ΓöÇΓöÇ noise attenuation ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            ctk.CTkLabel(c, text="Noise Attenuation",
                         font=ctk.CTkFont(size=13, weight="bold")).grid(
