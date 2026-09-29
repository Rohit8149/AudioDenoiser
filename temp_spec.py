                # Just center it roughly
                draw.text((IMG_W//2 - 60, IMG_H//2 - 5), text, fill="yellow")
            else:
                hist = self.dn.enh_hist if is_enh else self.dn.spec_hist
                if hist is None:
                    continue
                img = self._render_hist(hist, is_enh)
                
            photo = ImageTk.PhotoImage(img)
            lbl.configure(image=photo, width=IMG_W, height=IMG_H)
            lbl.image = photo

    # ═════════════════════════════════════════════════════════════════════
    #  EXIT
    # ═════════════════════════════════════════════════════════════════════

    def _exit(self):
        try:
            self._stop_all()
        finally:
            self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────
