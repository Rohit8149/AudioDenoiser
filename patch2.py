import re

with open('AudioDenoiser/audio_denoiser_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add self.profile_var to __init__
init_patch = '''        self.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        self._file_busy = False
        self.profile_var = tk.StringVar(value="Custom")'''
content = content.replace('''        self.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        self._file_busy = False''', init_patch)

# Remove the overwrite inside _open_settings
open_settings_patch = '''        ctk.CTkLabel(c, text="Where are you right now?:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w", pady=10)
        self.profile_sel = ctk.CTkComboBox(c, variable=self.profile_var, values=["Custom", "Traffic (Heavy Noise)", "Classroom (People Talking)", "Home (Normal)"], command=self._on_profile_change, width=250, state="readonly")'''
content = re.sub(r'        ctk\.CTkLabel\(c, text="Where are you right now\?:".*?state="readonly"\)', open_settings_patch, content, flags=re.DOTALL)


# Also ensure that _on_profile_change doesn't crash if atten_var isn't defined yet!
# If they haven't opened settings, atten_var won't exist. We should just update self.params["live"] directly!
on_profile_patch = '''    def _on_profile_change(self, choice: str):
        if choice == "Custom":
            self._broadcast_state()
            return
        
        # Define the presets: (atten, mix, gain, hpf, floor, post_filter)
        presets = {
            "Traffic (Heavy Noise)": (100, 100, 12, 160, 100, False),
            "Classroom (People Talking)": (100, 100, 12, 100, 100, True),
            "Home (Normal)": (60, 95, 12, 60, 40, False)
        }
        
        if choice in presets:
            atten, mix, gain, hpf, floor, pf = presets[choice]
            
            self._suppress_custom = True
            
            # Update internal params
            self.params["live"]["atten"] = atten
            self.params["live"]["mix"] = mix
            self.params["live"]["gain"] = gain
            self.params["live"]["hpf"] = hpf
            self.params["live"]["floor"] = floor
            self.pf = pf
            
            # Update Sliders if they exist
            if hasattr(self, "atten_var"):
                self.atten_var.set(atten)
                self.atten_lbl.configure(text=f"{atten} dB")
            
            if hasattr(self, "fx_lbls_live"):
                if "mix" in self.fx_lbls_live:
                    self.fx_lbls_live["mix"][0].set(mix)
                    self.fx_lbls_live["mix"][1].configure(text=self.fx_lbls_live["mix"][2](mix))
                if "gain" in self.fx_lbls_live:
                    self.fx_lbls_live["gain"][0].set(gain)
                    self.fx_lbls_live["gain"][1].configure(text=self.fx_lbls_live["gain"][2](gain))
                if "hpf" in self.fx_lbls_live:
                    self.fx_lbls_live["hpf"][0].set(hpf)
                    self.fx_lbls_live["hpf"][1].configure(text=self.fx_lbls_live["hpf"][2](hpf))
                if "floor" in self.fx_lbls_live:
                    self.fx_lbls_live["floor"][0].set(floor)
                    self.fx_lbls_live["floor"][1].configure(text=self.fx_lbls_live["floor"][2](floor))
                    
            if hasattr(self, "pf_var"):
                self.pf_var.set(pf)
                
            # Apply to engine
            if self.mode == "denoise" and self.dn is not None:
                self.dn.set_atten_lim(atten)
                self.dn.set_mix(mix / 100.0)
                self.dn.set_output_gain(gain)
                self.dn.set_highpass(hpf)
                self.dn.set_noise_floor(floor)
                self.dn.pf = pf
                
            self.root.after(100, lambda: setattr(self, "_suppress_custom", False))
            self._broadcast_state()'''

content = re.sub(r'    def _on_profile_change\(self, choice: str\):.*?self\._broadcast_state\(\)', on_profile_patch, content, flags=re.DOTALL)


with open('AudioDenoiser/audio_denoiser_app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch applied.")
