import sys
import time
import threading

def run_app():
    import audio_denoiser_app
    audio_denoiser_app.main()

t = threading.Thread(target=run_app, daemon=True)
t.start()

time.sleep(3)
print("Running app...")
