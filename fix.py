with open('AudioDenoiser/denoiser_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('self.biometric_threshold = 0.25', 'self.biometric_threshold = 0.20')

with open('AudioDenoiser/denoiser_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated pipeline baseline threshold!")
