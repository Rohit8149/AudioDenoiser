import os
import torch
import torchaudio
import numpy as np
from speechbrain.inference.separation import SepformerSeparation
from speaker_verifier import SpeakerVerifier

class MultiSpeakerSeparator:
    def __init__(self, device="cuda"):
        self.device = device
        
        # 1. Load the SepFormer Separation Model
        print("[SepFormer] Loading Cocktail Party Separation Model...")
        self.separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix",
            run_opts={"device": device}
        )
        
        # 2. Load our existing Biometric Verifier (ECAPA-TDNN)
        print("[SepFormer] Loading Biometric Voice Verifier...")
        self.verifier = SpeakerVerifier()

    def separate_and_isolate(self, mixed_audio_path, profile_path, chunk_duration_sec=8.0, progress_callback=None):
        """
        Takes a mixed audio file containing multiple overlapping speakers,
        separates them, uses the biometric profile to identify the target speaker,
        and stitches the target audio back together.
        """
        # Load the user's voice print
        self.verifier.load_profile(profile_path)
        
        # Load the overlapping mixed audio file
        mixed_sig, sr = torchaudio.load(mixed_audio_path)
        mixed_sig = mixed_sig.mean(dim=0, keepdim=True)  # Force mono
        
        # SepFormer wsj02mix processes at exactly 8000 Hz
        if sr != 8000:
            resampler_8k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)
            mixed_sig_8k = resampler_8k(mixed_sig)
        else:
            mixed_sig_8k = mixed_sig
            
        chunk_samples = int(chunk_duration_sec * 8000)
        total_samples = mixed_sig_8k.shape[1]
        
        final_audio_8k = []
        
        chunks_count = (total_samples + chunk_samples - 1) // chunk_samples
        
        resampler_16k_up = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
        
        for i, start in enumerate(range(0, total_samples, chunk_samples)):
            if progress_callback:
                progress_callback(i / chunks_count)
                
            end = min(start + chunk_samples, total_samples)
            chunk = mixed_sig_8k[:, start:end]
            
            # Skip tiny micro-chunks at the end of the file
            if chunk.shape[1] < 800: 
                final_audio_8k.append(chunk.squeeze(0))
                continue
            
            chunk = chunk.to(self.device)
            
            # Step A: Separate the overlapping voices into Track 1 and Track 2
            with torch.no_grad():
                est_sources = self.separator.separate_batch(chunk) 
                # Shape is [batch=1, time, sources=2]
            
            src1 = est_sources[0, :, 0].unsqueeze(0).cpu() # [1, time]
            src2 = est_sources[0, :, 1].unsqueeze(0).cpu()
            
            # Step B: Biometric Verification (ECAPA-TDNN needs 16kHz)
            src1_16k = resampler_16k_up(src1)
            src2_16k = resampler_16k_up(src2)
            
            score1 = self.verifier.verify_frame(src1_16k.numpy().flatten(), fs_in=16000)
            score2 = self.verifier.verify_frame(src2_16k.numpy().flatten(), fs_in=16000)
            
            # Step C: Select the track that matches the user
            if score1 > score2:
                winning_src = src1
                max_score = score1
            else:
                winning_src = src2
                max_score = score2
                
            # Step D: Gate. If neither track matches the user, mute.
            if max_score < 0.20:
                winning_src = torch.zeros_like(src1)
                
            final_audio_8k.append(winning_src.squeeze(0))
            
        # Stitch all the chunks back together
        stitched_8k = torch.cat(final_audio_8k, dim=0).unsqueeze(0)

        
        # Resample back up to 48000 Hz for the rest of the pipeline
        if sr != 8000:
            resampler_orig = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sr)
            final_audio = resampler_orig(stitched_8k)
        else:
            final_audio = stitched_8k
            
        return final_audio, sr

    def separate_and_isolate_array(self, audio_np, sr):
        """Processes a raw 1D numpy array in memory (for live streaming)."""
        mixed_sig = torch.from_numpy(audio_np).unsqueeze(0).float()
        
        if sr != 8000:
            resampler_8k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)
            mixed_sig_8k = resampler_8k(mixed_sig)
        else:
            mixed_sig_8k = mixed_sig
            
        chunk = mixed_sig_8k.to(self.device)
        with torch.no_grad():
            est_sources = self.separator.separate_batch(chunk)
            
        src1 = est_sources[0, :, 0].unsqueeze(0).cpu()
        src2 = est_sources[0, :, 1].unsqueeze(0).cpu()
        
        resampler_16k_up = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
        src1_16k = resampler_16k_up(src1)
        src2_16k = resampler_16k_up(src2)
        
        score1 = self.verifier.verify_frame(src1_16k.numpy().flatten(), fs_in=16000)
        score2 = self.verifier.verify_frame(src2_16k.numpy().flatten(), fs_in=16000)
        
        if score1 > score2:
            winning_src = src1
            max_score = score1
        else:
            winning_src = src2
            max_score = score2
            
        if max_score < 0.20:
            winning_src = torch.zeros_like(src1)
            
        if sr != 8000:
            resampler_orig = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sr)
            final_audio = resampler_orig(winning_src)
        else:
            final_audio = winning_src
            
        return final_audio.squeeze(0).numpy()
