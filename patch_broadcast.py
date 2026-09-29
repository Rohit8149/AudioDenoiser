import re

with open('AudioDenoiser/audio_denoiser_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_toggle_isolate = '''    def _toggle_isolate(self):
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
        self._broadcast_state()'''
        
content = re.sub(r'    def _toggle_isolate\(self\):.*?self\.dn\.reload_profile\(\)', new_toggle_isolate, content, flags=re.DOTALL)

new_toggle_cocktail = '''    def _toggle_cocktail(self):
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

content = re.sub(r'    def _toggle_cocktail\(self\):.*?text_color=COLOR_ON\)', new_toggle_cocktail, content, flags=re.DOTALL)

new_gain = '''        elif name == "gain":
            v = round(float(raw))
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_output_gain(v)
            self._broadcast_state()'''

content = content.replace('''        elif name == "gain":
            v = round(float(raw))
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_output_gain(v)''', new_gain)

with open('AudioDenoiser/audio_denoiser_app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch complete.")
