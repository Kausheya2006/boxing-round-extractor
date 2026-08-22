import os
import subprocess
import librosa
import torch
from transformers import ClapModel, ClapProcessor

from app.config import model_id, get_device
from app.utils import candidate_labels, sec_to_time, time_to_sec
from app.__init__ import *

device = get_device()

_processor = None
_model = None

def get_model(log_cb=None):
    global _processor, _model
    if _processor is None or _model is None:
        if log_cb: log_cb("Loading audio verification model... (this may take a moment)")
        
        # We will cache it inside the assets folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(base_dir, "assets", "models")
        os.makedirs(cache_dir, exist_ok=True)
        
        _processor = ClapProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        _model = ClapModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
        _model.eval()
        
        if log_cb: log_cb("Model loaded successfully.")
    return _processor, _model

def predict(audio_path, verbose=False, log_cb=None):
    if os.getenv("MOCK_MODE") == "True":
        if log_cb: log_cb(f"MOCK_MODE enabled: Mocking audio verification for {audio_path}")
        return True # Mock success

    try:
        processor, model = get_model(log_cb=log_cb)
        
        waveform, sr = librosa.load(audio_path, sr=48000)
        inputs = processor(
            text=candidate_labels, 
            audio=waveform, 
            return_tensors="pt", 
            padding=True,
            sampling_rate=48000
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_audio.softmax(dim=-1)[0]

        if log_cb:
            log_cb(f"Audio Class Probabilities:")
            for label, prob in zip(candidate_labels, probs):
                log_cb(f"  {prob * 100:>5.1f}% : {label}")
                
        if verbose:
            for label, prob in zip(candidate_labels, probs):
                    print(f"{prob * 100:>5.1f}% : {label}")
                    
        return probs[0].item() > 0.15
    except Exception as e:
        if log_cb: log_cb(f"Error in prediction: {e}")
        return False





def verify_rounds(video_path, round_extract_lists, log_cb=None, check_cancel=None):
    from pathlib import Path
    video_stem = Path(video_path).stem
    temp_dir = f"assets/temp_audio_{video_stem}"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    if log_cb: log_cb(f"Starting verification of {len(round_extract_lists)} timestamps...")
    verified_timestamps = []
    last_event_sec = -999
    
    # We will test offsets from -3 to +3 seconds
    offsets = [-5,-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    for idx, t_str in enumerate(round_extract_lists):
        round_num = (idx // 2) + 1
        is_start = idx % 2 == 0
        stage = "Start" if is_start else "End"
        
        t_sec = time_to_sec(t_str)
        
        msg = f"\nVerifying Round {round_num} {stage} at {t_str}"
        if log_cb: log_cb(msg)
        else: print(msg)
        
        predictions = []
        temp_files = []

        for offset in offsets:
            if check_cancel and check_cancel():
                if log_cb: log_cb("Audio verification cancelled by user.")
                raise Exception("Cancelled by user")

            start_time = t_sec + offset
            if start_time < 0: start_time = 0
            
            # Prevent intersection with the previous bell (a bell rings for ~3 seconds)
            if idx > 0 and start_time <= last_event_sec + 3:
                predictions.append(False)
                continue

            temp_audio = f"{temp_dir}/temp_window_{idx}_{offset}.wav"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            temp_files.append(temp_audio)
            
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-t", "1", 
                "-i", video_path, "-vn", "-ar", "48000", "-ac", "1", 
                temp_audio, "-loglevel", "quiet"
            ]
            subprocess.run(cmd)

            is_bell = predict(temp_audio, log_cb=log_cb)
            predictions.append(is_bell)
            
            # if os.path.exists(temp_audio):
            #     os.remove(temp_audio)

        if log_cb: log_cb(f"   Audio Array: {predictions}")
        else: print(f"   Audio Array: {predictions}")
        
        verified_sec = t_sec # Default to the OCR anchor
        is_verified_by_audio = False

        # --- FIRST OCCURRENCE LOGIC ---
        for offset, is_bell in zip(offsets, predictions):
            if is_bell:
                verified_sec = t_sec + offset
                is_verified_by_audio = True
                if offset != 0:
                    msg = f"Audio Confirmed! First True at offset {offset:+d}s -> Adjusted to: {sec_to_time(verified_sec)}"
                    if log_cb: log_cb(msg)
                    else: print(msg)
                break # Lock in the very first hit and stop searching
        
        if not is_verified_by_audio:
            msg = f"No audio trigger found. Falling back to OCR anchor: {sec_to_time(verified_sec)}"
            if log_cb: log_cb(msg)
            else: print(msg)
            verified_timestamps.append(f"{sec_to_time(verified_sec)} (Unverified)")
        else:
            verified_timestamps.append(sec_to_time(verified_sec))
            
        last_event_sec = verified_sec

    return verified_timestamps
