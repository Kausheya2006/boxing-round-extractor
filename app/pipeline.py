from app.extract.round_extract import RoundExtractor
from app.__init__ import *
from app.extract.verifier import verify_rounds

def extract_rounds_from_video(video_path: str, reuse_debug_log=False, log_cb=None, check_cancel=None) -> dict:
    extractor = RoundExtractor(video_path)
    result = extractor.run(num_samples=250, reuse_debug_log=reuse_debug_log, log_cb=log_cb, check_cancel=check_cancel)
    
    if not result:
        raise Exception("Failed to extract rounds. Ensure the video is a valid boxing match.")
        
    rounds, round_length, total_rounds = result
    
    return {
        "extrapolated_rounds": rounds,
        "round_length": round_length,
        "total_rounds": total_rounds
    }

def verify_extracted_rounds(video_path: str, extrapolated_rounds: list, log_cb=None, check_cancel=None) -> list:
    results = verify_rounds(video_path, extrapolated_rounds, log_cb=log_cb, check_cancel=check_cancel)
    return results