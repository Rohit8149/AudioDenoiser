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
                self.tip_lbl.configure(text="≡ƒô₧ Teams Mode ON: AI audio is now routed to Virtual Cable.", text_color=COLOR_ON)
            else:
                self.teams_var.set(False)
                self.tip_lbl.configure(text="ΓÜá∩╕Å VB-Audio Virtual Cable not found! Please install it.", text_color="red")
        else:
            if hasattr(self, "_prev_out_dev"):
                self.out_dev.set(self._prev_out_dev)
            self.tip_lbl.configure(text="≡ƒô₧ Teams Mode OFF: Audio routed back to normal speakers.", text_color=COLOR_ON)
            
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
                self.tip_lbl.configure(text="ΓÅ│ Loading massive SepFormer AI into memory (takes a few secs)...", text_color="yellow")
                self.root.update_idletasks()
                
                def load_task():
                    try:
                        self.dn.set_cocktail_mode(True)
                        self.root.after(0, lambda: self.live_cocktail_switch.configure(text="Cocktail Party Separation (3s Delay)", state="normal"))
                        self.root.after(0, lambda: self.tip_lbl.configure(text="Γ£à Cocktail AI Active! (Expect 3s audio delay)", text_color=COLOR_ON))
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
            self.enroll_prompt.configure(text="ΓÜá∩╕Å Please select an output device first!")
            return
            
        self.play_voice_btn.configure(state="disabled")
        self.enroll_prompt.configure(text="Γû╢∩╕Å Playing your saved Voice Print...", text_color=COLOR_ON)
        
        def play_task():
            try:
                import soundfile as sf
                data, fs = sf.read(profile_path)
                sd.play(data, fs, device=dev_out)
                sd.wait()
                self.enroll_prompt.configure(text="Γ£à Finished playing back your Voice Print.", text_color=COLOR_DIM)
            except Exception as e:
                self.enroll_prompt.configure(text=f"Γ¥î Error playing: {e}", text_color=COLOR_OFF)
            finally:
                self.play_voice_btn.configure(state="normal")
                
        threading.Thread(target=play_task, daemon=True).start()

    def _enroll_voice(self):
        try:
            dev_in = int(self.in_dev.get().split(":")[0])
        except (ValueError, IndexError):
            self.enroll_prompt.configure(text="Γ¥î Please select a microphone first!")
            return

        self.enroll_btn.configure(state="disabled")
        self.play_voice_btn.configure(state="disabled")
        self.enroll_prompt.configure(text="≡ƒö┤ RECORDING (20s)...", text_color="orange")
        
        abort_flag = [False]
        
        # --- Teleprompter Overlay ---
        prompt_win = ctk.CTkToplevel(self.root)
        
        def on_close_prompt():
            abort_flag[0] = True
            import sounddevice as sd
            sd.stop()
            prompt_win.destroy()
            self.enroll_prompt.configure(text="Γ¥î Enrollment aborted by user.", text_color="#FF4444")
            self.enroll_btn.configure(state="normal", text="≡ƒöä Re-enroll Voice")
            if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")):
                self.play_voice_btn.configure(state="normal")
                self.isolate_switch.configure(state="normal")

        prompt_win.protocol("WM_DELETE_WINDOW", on_close_prompt)
        prompt_win.title("Voice Enrollment")
        prompt_win.geometry("500x350")
        prompt_win.attributes("-topmost", True)
        prompt_win.geometry(f"+{self.root.winfo_x() + 100}+{self.root.winfo_y() + 50}")
        
        lbl_title = ctk.CTkLabel(prompt_win, text="≡ƒö┤ RECORDING IN PROGRESS", text_color="#FF4444", font=ctk.CTkFont(size=20, weight="bold"))
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
                
                self.enroll_prompt.configure(text="≡ƒº╣ Purifying voice print using DeepFilterNet...", text_color="yellow")
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
                
                self.enroll_prompt.configure(text="Γ£à Voice Print saved! The AI will now use this to filter out other voices.", text_color=COLOR_ON)
                self.enroll_btn.configure(text="≡ƒöä Re-enroll Voice")
            except Exception as e:
                self.enroll_prompt.configure(text=f"Γ¥î Error recording: {e}", text_color=COLOR_OFF)
            finally:
                self.enroll_btn.configure(state="normal")
                if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speaker_profile.wav")):
                    self.play_voice_btn.configure(state="normal")
                    self.isolate_switch.configure(state="normal")
                
                
        threading.Thread(target=record_task, daemon=True).start()

