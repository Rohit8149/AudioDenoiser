import os
import torch
import torchaudio
import numpy as np
from speechbrain.inference.separation import SepformerSeparation
from speaker_verifier import SpeakerVerifier

class MultiSpeakerSeparator:
    def __init__(self, device="cuda"):
        self.device = device
        
        # 1. Load the SepFormer 16kHz Separation Model
        print("[SepFormer] Loading 16kHz Cocktail Party Separation Model...")
        self.separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir="pretrained_models/sepformer-whamr16k",
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
        
        # New Model processes at exactly 16000 Hz
        if sr != 16000:
            resampler_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            mixed_sig_16k = resampler_16k(mixed_sig)
        else:
            mixed_sig_16k = mixed_sig
            
        chunk_samples = int(chunk_duration_sec * 16000)
        total_samples = mixed_sig_16k.shape[1]
        
        final_audio_16k = []
        
        chunks_count = (total_samples + chunk_samples - 1) // chunk_samples
        
        for i, start in enumerate(range(0, total_samples, chunk_samples)):
            if progress_callback:
                progress_callback(i / chunks_count)
                
            end = min(start + chunk_samples, total_samples)
            chunk = mixed_sig_16k[:, start:end]
            
            # Skip tiny micro-chunks at the end of the file
            if chunk.shape[1] < 1600: 
                final_audio_16k.append(chunk.squeeze(0))
                continue
            
            chunk = chunk.to(self.device)
            
            # Step A: Separate the overlapping voices into Track 1 and Track 2
            with torch.no_grad():
                est_sources = self.separator.separate_batch(chunk) 
                # Shape is [batch=1, time, sources=2]
            
            src1 = est_sources[0, :, 0].unsqueeze(0).cpu() # [1, time]
            src2 = est_sources[0, :, 1].unsqueeze(0).cpu()
            
            # Step B: Biometric Verification (ECAPA-TDNN natively uses 16kHz, so no resampling needed!)
            score1 = self.verifier.verify_frame(src1.numpy().flatten(), fs_in=16000)
            score2 = self.verifier.verify_frame(src2.numpy().flatten(), fs_in=16000)
            
            # Step C: Select the track that matches the user
            if score1 > score2:
                winning_src = src1
                max_score = score1
            else:
                winning_src = src2
                max_score = score2
                
            print(f"\n[Offline Cocktail] Chunk {i+1} | Track 1: {score1*100:.1f}% | Track 2: {score2*100:.1f}%")
            if max_score >= threshold:
                print(f" -> PASSED: Identity matched with {max_score*100:.1f}%.")
            else:
                print(f" -> BLOCKED: Highest similarity ({max_score*100:.1f}%) is below 25.0%.")
                
            # Step D: Gate. Threshold set to 0.25 to securely block YouTube without rejecting you
            if max_score < threshold:
                winning_src = torch.zeros_like(src1)
            else:
                # Restore natural volume! (SepFormer aggressively amplifies quiet audio)
                orig_rms = torch.sqrt(torch.mean(chunk.cpu()**2))
                est_rms = torch.sqrt(torch.mean(winning_src**2))
                if est_rms > 0.0001:
                    winning_src = winning_src * (orig_rms / est_rms)
                
            final_audio_16k.append(winning_src.squeeze(0))
            
        # Stitch all the chunks back together
        stitched_16k = torch.cat(final_audio_16k, dim=0).unsqueeze(0)

        # Resample back up to original SR (48000) for the rest of the pipeline
        if sr != 16000:
            resampler_orig = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sr)
            final_audio = resampler_orig(stitched_16k)
        else:
            final_audio = stitched_16k
            
        return final_audio, sr

    def separate_and_isolate_array(self, audio_np, sr, threshold=0.25):
        """Processes a raw 1D numpy array in memory (for live streaming)."""
        mixed_sig = torch.from_numpy(audio_np).unsqueeze(0).float()
        
        if sr != 16000:
            resampler_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            mixed_sig_16k = resampler_16k(mixed_sig)
        else:
            mixed_sig_16k = mixed_sig
            
        chunk = mixed_sig_16k.to(self.device)
        with torch.no_grad():
            est_sources = self.separator.separate_batch(chunk)
            
        src1 = est_sources[0, :, 0].unsqueeze(0).cpu()
        src2 = est_sources[0, :, 1].unsqueeze(0).cpu()
        
        # ECAPA-TDNN expects 16kHz, which matches our new SepFormer perfectly!
        score1 = self.verifier.verify_frame(src1.numpy().flatten(), fs_in=16000)
        score2 = self.verifier.verify_frame(src2.numpy().flatten(), fs_in=16000)
        
        if score1 > score2:
            winning_src = src1
            max_score = score1
        else:
            winning_src = src2
            max_score = score2
            
        # --- Live Diagnostics for the Terminal ---
        print(f"\n[Cocktail Mode] Biometric Similarity | Track 1: {score1*100:.1f}% | Track 2: {score2*100:.1f}%")
        if max_score >= threshold:
            print(f" -> PASSED: Your voice was identified on Track {1 if score1 > score2 else 2} with {max_score*100:.1f}% match.")
        else:
            print(f" -> BLOCKED: VIP Gate failed. Highest match was {max_score*100:.1f}%. (Threshold is {threshold*100:.1f}%).")
        # -----------------------------------------
            
        # Raised threshold to 0.25! (0.40 was too strict and rejected the legitimate user)
        if max_score < threshold:
            winning_src = torch.zeros_like(src1)
        else:
            # Restore natural volume! (SepFormer aggressively amplifies quiet audio)
            orig_rms = torch.sqrt(torch.mean(chunk.cpu()**2))
            est_rms = torch.sqrt(torch.mean(winning_src**2))
            if est_rms > 0.0001:
                winning_src = winning_src * (orig_rms / est_rms)
            
        if sr != 16000:
            resampler_orig = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sr)
            final_audio = resampler_orig(winning_src)
        else:
            final_audio = winning_src
            
        return final_audio.squeeze(0).numpy(), max_score
