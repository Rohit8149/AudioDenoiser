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