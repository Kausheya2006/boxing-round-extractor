from .__init__ import *
import re

from app.config import VERIFY_MOMENTS_MODEL, VERIFY_MOMENTS_PROMPT

class VerifyMomentsConfig:
    def __init__(self, prompt: str, model_name: str = VERIFY_MOMENTS_MODEL, fps: float = 16.0):
        self.model_name = model_name
        self.prompt = prompt
        self.fps = fps
        self.generation_config = {
            'max_output_tokens': 65536,
            'temperature': 0.1
        }

class VerifyMomentsParser:
    @staticmethod
    def extract(markdown_text: str):
        if not markdown_text:
            return {"verified": False}
            
        # Try to find the strict bracket output, e.g., [YES, "Knockdown", "Clean", "Red vs Blue", "Left Hook", "Fell"]
        match = re.search(r'\[(YES|NO)(.*?)\]', markdown_text, re.IGNORECASE | re.DOTALL)
        if not match:
            # Maybe the whole text is the list
            match = re.search(r'\[\s*(YES|NO).*?\]', markdown_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            status = match.group(1).upper()
            if status == "NO":
                return {"verified": False}
            else:
                rest = match.group(2)
                # Split by comma but respect quotes
                # Simple heuristic: split by comma, clean quotes
                parts = []
                current_part = ""
                in_quotes = False
                for char in rest:
                    if char == '"' or char == "'":
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        parts.append(current_part.strip(' \n"\''))
                        current_part = ""
                    else:
                        current_part += char
                parts.append(current_part.strip(' \n"\''))
                
                parts = [p for p in parts if p] # filter empty
                
                # We expect 5 parts after YES
                category = parts[0] if len(parts) > 0 else "Unknown"
                landed_status = parts[1] if len(parts) > 1 else "Unknown"
                attacker_vs_defender = parts[2] if len(parts) > 2 else "Unknown"
                attack_type = parts[3] if len(parts) > 3 else "Unknown"
                defender_reaction = parts[4] if len(parts) > 4 else "Unknown"
                
                return {
                    "verified": True,
                    "category": category,
                    "landed_status": landed_status,
                    "attacker_vs_defender": attacker_vs_defender,
                    "attack_type": attack_type,
                    "defender_reaction": defender_reaction
                }
        
        return {"verified": False}


class GeminiClipVerifier:
    def __init__(self, api_key: str, config: VerifyMomentsConfig, log_cb=None):
        self.client = genai.Client(api_key=api_key)
        self.config = config
        self.log_cb = log_cb

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)
        else:
            print(msg)

    def upload_and_wait(self, video_path: str):
        video_file = self.client.files.upload(file=video_path)
        while hasattr(video_file, 'state') and str(video_file.state) == "PROCESSING":
            time.sleep(2)
            video_file = self.client.files.get(name=video_file.name)
            
        if hasattr(video_file, 'state') and str(video_file.state) == "FAILED":
            raise ValueError(f"Video processing failed for {video_file.name}")
            
        return video_file

    def run_session(self, video_file) -> str:
        video_part = types.Part(
            file_data=types.FileData(
                file_uri=video_file.uri, 
                mime_type=video_file.mime_type
            ),
            video_metadata=types.VideoMetadata(fps=self.config.fps) 
        )
        
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=[video_part, self.config.prompt],
            config=self.config.generation_config,
        )
        return response.text

    def delete_file(self, file_name: str):
        try:
            self.client.files.delete(name=file_name)
        except Exception as e:
            self._log(f"Warning: Could not delete video file automatically: {e}")

def time_to_sec(time_str):
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

def verify_key_moments(video_path: str, moments_dict: dict, log_cb=None, check_cancel=None) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in .env")

    config = VerifyMomentsConfig(prompt=VERIFY_MOMENTS_PROMPT, fps=16.0)
    verifier = GeminiClipVerifier(api_key=api_key, config=config, log_cb=log_cb)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    video_stem = Path(video_path).stem
    temp_dir = os.path.join(base_dir, "assets", f"temp_clips_{video_stem}", "verify_clips")
    os.makedirs(temp_dir, exist_ok=True)

    verified_moments = {}

    for round_name, timestamps in moments_dict.items():
        if check_cancel and check_cancel():
            if log_cb: log_cb("Verify Moments cancelled by user.")
            raise Exception("Cancelled by user")
            
        verified_moments[round_name] = []
        
        if not timestamps:
            continue

        if log_cb: log_cb(f"\n--- Visually Verifying {len(timestamps)} moments in {round_name} ---")
        
        for ts in timestamps:
            if check_cancel and check_cancel():
                raise Exception("Cancelled by user")

            t_sec = time_to_sec(ts)
            
            # Extract exactly 2 seconds. We will start 0.5s before the timestamp to catch the windup
            start_sec = max(0, t_sec - 0.5)
            duration = 2.0
            
            clip_path = os.path.join(temp_dir, f"{round_name.replace(' ', '_')}_{ts.replace(':', '_')}.mp4")
            
            if not os.path.exists(clip_path):
                cmd = [
                    "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration),
                    "-i", video_path, "-c", "copy", clip_path, "-loglevel", "quiet"
                ]
                subprocess.run(cmd, check=True)

            if os.getenv("MOCK_MODE") == "True":
                if log_cb: log_cb(f"[MOCK MODE] Simulating verification for {ts}")
                time.sleep(1)
                result_data = {
                    "verified": True,
                    "category": "Knockdown",
                    "landed_status": "Cleanly Landed",
                    "attacker_vs_defender": "Red vs Blue",
                    "attack_type": "Right Cross",
                    "defender_reaction": "Fell to canvas"
                }
            else:
                if log_cb: log_cb(f"Uploading clip for {ts}...")
                video_file = verifier.upload_and_wait(clip_path)
                if log_cb: log_cb(f"Analyzing clip for {ts}...")
                try:
                    res_text = verifier.run_session(video_file)
                    
                    # Log the response
                    log_file = os.path.join(temp_dir, f"{round_name.replace(' ', '_')}_{ts.replace(':', '_')}_response.md")
                    with open(log_file, "w") as f:
                        f.write(res_text)
                        
                    result_data = VerifyMomentsParser.extract(res_text)
                except Exception as e:
                    if log_cb: log_cb(f"Error during verification: {e}")
                    result_data = {"verified": False}
                finally:
                    verifier.delete_file(video_file.name)
            
            result_data["timestamp"] = ts
            verified_moments[round_name].append(result_data)
            
            status_str = "Verified YES" if result_data["verified"] else "Rejected NO"
            if log_cb: log_cb(f"Result for {ts}: {status_str}")

    if log_cb: log_cb("\nCompleted visual verification.")
    return verified_moments
