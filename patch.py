with open('AudioDenoiser/audio_denoiser_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. _toggle_isolate
old_iso = '''    def _toggle_isolate(self):
        if self.isolate_var.get() and getattr(self, "live_cocktail_var", None) and self.live_cocktail_var.get():
            self.live_cocktail_var.set(False)
            self._toggle_cocktail()
            
        if self.dn is not None:
            self.dn.isolate_speaker = self.isolate_var.get()
            if self.isolate_var.get():
                if hasattr(self, "threshold_var"):
                    self.threshold_var.set(0.20)
                    self._on_thresh_change(0.20)
                self.dn.reload_profile()'''
                
new_iso = old_iso + '\n        self._broadcast_state()'
content = content.replace(old_iso, new_iso)

# 2. _toggle_cocktail
old_cock = '''    def _toggle_cocktail(self):
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
                self.tip_lbl.configure(text="Cocktail Mode OFF. Standard Denoising Active.", text_color=COLOR_ON)'''

new_cock = '''    def _toggle_cocktail(self):
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
                        self.root.after(0, self._broadcast_state)
                    except Exception as e:
                        log(f"Failed to load cocktail mode: {e!r}")
                        self.root.after(0, lambda: self.live_cocktail_var.set(False))
                        self.root.after(0, lambda: self.live_cocktail_switch.configure(text="Cocktail Party Separation (3s Delay)", state="normal"))
                        self.root.after(0, self._broadcast_state)
                
                import threading
                threading.Thread(target=load_task, daemon=True).start()
            else:
                self.dn.set_cocktail_mode(False)
                self.tip_lbl.configure(text="Cocktail Mode OFF. Standard Denoising Active.", text_color=COLOR_ON)
                self._broadcast_state()'''

content = content.replace(old_cock, new_cock)

# 3. gain param
old_gain = '''        elif name == "gain":
            v = round(float(raw))
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_output_gain(v)'''
new_gain = old_gain + '\n            self._broadcast_state()'
content = content.replace(old_gain, new_gain)

with open('AudioDenoiser/audio_denoiser_app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done patching.")
