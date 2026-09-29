with open('AudioDenoiser/audio_denoiser_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix gain missing parameter save
old_gain = '''        elif name == "gain":
            v = round(float(raw))
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_output_gain(v)
            self._broadcast_state()'''

new_gain = '''        elif name == "gain":
            v = round(float(raw))
            self.params[tab][name] = v
            lbl.configure(text=fmt(v))
            if self.mode == "denoise":
                self.dn.set_output_gain(v)
            self._broadcast_state()'''
            
content = content.replace(old_gain, new_gain)

# mix
old_mix = '''        if name == "mix":
            v = int(round(float(raw)))
            lbl.configure(text=fmt(v))'''
new_mix = '''        if name == "mix":
            v = int(round(float(raw)))
            self.params[tab][name] = v
            lbl.configure(text=fmt(v))'''
content = content.replace(old_mix, new_mix)

# hpf
old_hpf = '''        elif name == "hpf":
            v = round(float(raw) / 20) * 20
            lbl.configure(text=fmt(v))'''
new_hpf = '''        elif name == "hpf":
            v = round(float(raw) / 20) * 20
            self.params[tab][name] = v
            lbl.configure(text=fmt(v))'''
content = content.replace(old_hpf, new_hpf)

# floor
old_floor = '''        elif name == "floor":
            v = int(round(float(raw)))
            lbl.configure(text=fmt(v))'''
new_floor = '''        elif name == "floor":
            v = int(round(float(raw)))
            self.params[tab][name] = v
            lbl.configure(text=fmt(v))'''
content = content.replace(old_floor, new_floor)

with open('AudioDenoiser/audio_denoiser_app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Parameters patched.")
