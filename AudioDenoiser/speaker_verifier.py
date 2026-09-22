import os
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# Windows symlink fallback patch (fixes OSError 1314 when downloading models without Developer Mode)
import pathlib
import shutil
_orig_symlink = pathlib.Path.symlink_to
def _safe_symlink_to(self, target, target_is_directory=False):
    try:
        _orig_symlink(self, target, target_is_directory)
    except OSError:
        if self.exists():
            self.unlink()
        if os.path.isdir(target):
            shutil.copytree(target, self)
        else:
            shutil.copy2(target, self)
pathlib.Path.symlink_to = _safe_symlink_to

class SpeakerVerifier:
    """
    Handles speaker verification using SpeechBrain's ECAPA-TDNN model.
    It can enroll a voice from a wav file and verify if live audio matches the profile.
    """
    def __init__(self, savedir="pretrained_models/spkrec-ecapa-voxceleb"):
        # We explicitly load to CPU to avoid CUDA overhead/memory issues during fast streaming
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"}
        )
        self.target_embedding = None
        self.resampler = torchaudio.transforms.Resample(48000, 16000)

    def load_profile(self, wav_path):
        """Loads a voice profile and computes its d-vector embedding."""
        if not os.path.exists(wav_path):
            self.target_embedding = None
            return False
            
        try:
            signal, fs = torchaudio.load(wav_path)
            
            # Ensure mono
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz (what ECAPA expects)
            if fs != 16000:
                signal = torchaudio.transforms.Resample(fs, 16000)(signal)
            
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(signal)
                self.target_embedding = embeddings.squeeze() # Flatten to 1D
            return True
        except Exception as e:
            print(f"Error loading voice profile: {e}")
            self.target_embedding = None
            return False

    def verify_frame(self, block, fs_in=48000):
        """
        Takes a block of audio (numpy array, shape [N]), returns a similarity score [-1, 1].
        If no profile is enrolled, returns 1.0 (pass all).
        """
        if self.target_embedding is None:
            return 1.0
            
        try:
            # Convert numpy array to torch tensor [1, N]
            signal = torch.from_numpy(block).unsqueeze(0).to(torch.float32)
            
            # Fast resampling using cached resampler
            if fs_in == 48000:
                signal = self.resampler(signal)
            elif fs_in != 16000:
                signal = torchaudio.transforms.Resample(fs_in, 16000)(signal)
            
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(signal).squeeze()
                
            # Cosine similarity between target embedding and current frame embedding
            score = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(0), 
                self.target_embedding.unsqueeze(0)
            )
            return score.item()
        except Exception as e:
            print(f"Verify error: {e}")
            return 1.0
