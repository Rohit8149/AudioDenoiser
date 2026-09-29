with open('AudioDenoiser/multi_speaker_separator.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# We will use regex to safely replace the logic!
# First replace the method signature if it isn't already updated
content = re.sub(r'def separate_and_isolate_array\(self, audio_np, sr\):', 
                 r'def separate_and_isolate_array(self, audio_np, sr, threshold=0.25):', 
                 content)

# Now safely replace the hardcoded 0.25 threshold inside the print and logic
content = re.sub(r'if max_score >= 0\.25:', r'if max_score >= threshold:', content)
content = re.sub(r'Threshold is 25\.0%', r'Threshold is {threshold*100:.1f}%', content)
content = re.sub(r'if max_score < 0\.25:', r'if max_score < threshold:', content)

# Finally, change the return statement to return BOTH final_audio and max_score
content = re.sub(r'return final_audio\.squeeze\(0\)\.numpy\(\)', 
                 r'return final_audio.squeeze(0).numpy(), max_score', 
                 content)

with open('AudioDenoiser/multi_speaker_separator.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated Cocktail Separator correctly!")
