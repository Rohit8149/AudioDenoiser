import subprocess
import time

p = subprocess.Popen([r"AudioDenoiser\run.bat"])
time.sleep(3)
p.terminate()
