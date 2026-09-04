from .__init__ import *

from app.config import GEMINI_MODEL, MOMENTS_PROMPT

class AnalyzerConfig:
    def __init__(self, prompt: str, model_name: str = GEMINI_MODEL, fps: float = 1.0):
        self.model_name = model_name
        self.prompt = prompt
        self.fps = fps
        self.generation_config = {
            'max_output_tokens': 65536,
            'temperature': 1.0
        }

class TimestampParser:
    @staticmethod
    def extract(markdown_text: str) -> List[str]:
        timestamps = []
        if not markdown_text:
            return timestamps
            
        lines = markdown_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- [') and ']' in line:
                # e.g., - [01:24] Event: Knockdown ...
                end_bracket = line.find(']')
                ts = line[3:end_bracket]
                timestamps.append(ts)
        return timestamps

    @staticmethod
    def merge_and_sort(list1: List[str], list2: List[str]) -> List[str]:
        final_union = set(list1).union(set(list2))
        return sorted(list(final_union))

class GeminiVideoAnalyzer:
    def __init__(self, api_key: str, config: AnalyzerConfig, log_cb=None):
        self.client = genai.Client(api_key=api_key)
        self.config = config
        self.log_cb = log_cb

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)
        else:
            print(msg)

    def upload_and_wait(self, video_path: str):
        self._log(f"Uploading {video_path}...")
        video_file = self.client.files.upload(file=video_path)
        self._log(f"Uploaded as {video_file.name}. Waiting for processing...")
        
        while hasattr(video_file, 'state') and str(video_file.state) == "PROCESSING":
            time.sleep(5)
            video_file = self.client.files.get(name=video_file.name)
            
        if hasattr(video_file, 'state') and str(video_file.state) == "FAILED":
            raise ValueError(f"Video processing failed for {video_file.name}")
            
        self._log("Video is ready for analysis.")
        return video_file

    def run_session(self, video_file, session_id: int) -> str:
        self._log(f"Starting Session {session_id}...")
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
            self._log("Video file deleted from API server.")
        except Exception as e:
            self._log(f"Warning: Could not delete video file automatically: {e}")

    def process_video(self, video_path: str) -> List[str]:
        if os.getenv("MOCK_MODE") == "True":
            self._log(f"[MOCK MODE] Simulating Gemini for {video_path}...")
            time.sleep(2)
            mock_markdown = "- [01:10] Event: Power Punch\n- [02:14] Event: Knockdown"
            timestamps_1 = TimestampParser.extract(mock_markdown)
            return TimestampParser.merge_and_sort(timestamps_1, timestamps_1)

        video_file = self.upload_and_wait(video_path)
        outputs = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_session = {
                executor.submit(self.run_session, video_file, 1): 1, 
                executor.submit(self.run_session, video_file, 2): 2 
            }
            for future in as_completed(future_to_session):
                try:
                    res_text = future.result()
                    outputs.append(res_text)
                    
                    # Save the raw response for logging
                    session_id = future_to_session[future]
                    video_stem = Path(video_path).stem
                    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", f"gemini_responses_{video_stem}")
                    os.makedirs(log_dir, exist_ok=True)
                    log_file = os.path.join(log_dir, f"session_{session_id}.md")
                    
                    with open(log_file, "a") as f:
                        f.write(f"\n--- Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                        f.write(res_text)
                        f.write("\n")
                        
                except Exception as exc:
                    self._log(f"Session generated an exception: {exc}")

        self.delete_file(video_file.name)

        if len(outputs) == 2:
            timestamps_1 = TimestampParser.extract(outputs[0])
            timestamps_2 = TimestampParser.extract(outputs[1])
            self._log(f"Session 1 found: {timestamps_1}")
            self._log(f"Session 2 found: {timestamps_2}")
            return TimestampParser.merge_and_sort(timestamps_1, timestamps_2)
        else:
            self._log("Failed to get outputs from both sessions.")
            return []

def time_to_sec(time_str):
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

def format_sec(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def map_timestamp_to_global(clip_ts: str, round_start_sec: int) -> str:
    parts = [int(p) for p in clip_ts.split(':')]
    total_sec = parts[0] * 60 + parts[1]
    total_sec += round_start_sec
    return format_sec(total_sec)

def find_key_moments_for_rounds(video_path: str, rounds_list: list, log_cb=None, check_cancel=None) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in .env")

    config = AnalyzerConfig(prompt=MOMENTS_PROMPT, fps=1.0)
    analyzer = GeminiVideoAnalyzer(api_key=api_key, config=config, log_cb=log_cb)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    video_stem = Path(video_path).stem
    temp_dir = os.path.join(base_dir, "assets", f"temp_clips_{video_stem}")
    os.makedirs(temp_dir, exist_ok=True)

    all_moments = {}
    
    # We define a helper within this function for the Gemini call
    # since we want to handle the response saving differently now.
    
    # rounds_list is [start1, end1, start2, end2, ...]
    for i in range(0, len(rounds_list), 2):
        if check_cancel and check_cancel():
            if log_cb: log_cb("Find Moments cancelled by user.")
            raise Exception("Cancelled by user")

        round_num = (i // 2) + 1
        start_str = rounds_list[i].replace(" (Unverified)", "")
        end_str = rounds_list[i+1].replace(" (Unverified)", "")
        start_sec = time_to_sec(start_str)
        end_sec = time_to_sec(end_str)
        duration = end_sec - start_sec

        if duration <= 0:
            if log_cb: log_cb(f"Skipping Round {round_num} due to invalid duration ({duration}s).")
            continue

        if log_cb: log_cb(f"\n--- Processing Round {round_num} for Key Moments ---")
        
        round_dir = os.path.join(base_dir, "assets", video_stem, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        responses_file = os.path.join(round_dir, "responses.md")
        
        # Check round-level cache
        if os.path.exists(responses_file):
            if log_cb: log_cb(f"Found cached responses for Round {round_num}. Parsing directly...")
            with open(responses_file, "r") as f:
                cached_responses = f.read()
            local_timestamps = TimestampParser.extract(cached_responses)
            # Remove duplicates by merging with itself
            local_timestamps = TimestampParser.merge_and_sort(local_timestamps, local_timestamps)
        else:
            # Need to call Gemini
            clip_path = os.path.join(temp_dir, f"round_{round_num}.mp4")
            
            # Clip the video
            if not os.path.exists(clip_path):
                cmd = [
                    "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration),
                    "-i", video_path, "-c", "copy", clip_path, "-loglevel", "quiet"
                ]
                subprocess.run(cmd, check=True)

            # Process with Gemini natively (no _log saving inside process_video)
            # We'll re-implement the concurrent processing here to capture the raw strings together.
            if os.getenv("MOCK_MODE") == "True":
                if log_cb: log_cb(f"[MOCK MODE] Simulating Gemini for {clip_path}...")
                time.sleep(2)
                mock_text = "- [01:10] Event: Power Punch\n- [02:14] Event: Knockdown"
                outputs = [mock_text, mock_text]
            else:
                video_file = analyzer.upload_and_wait(clip_path)
                outputs = []
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_session = {
                        executor.submit(analyzer.run_session, video_file, 1): 1, 
                        executor.submit(analyzer.run_session, video_file, 2): 2 
                    }
                    for future in as_completed(future_to_session):
                        try:
                            outputs.append(future.result())
                        except Exception as exc:
                            if log_cb: log_cb(f"Session generated an exception: {exc}")
                analyzer.delete_file(video_file.name)

            if len(outputs) > 0:
                # Save to responses.md
                with open(responses_file, "w") as f:
                    for idx, out_text in enumerate(outputs):
                        f.write(f"--- Session {idx+1} ---\n")
                        f.write(out_text)
                        f.write("\n\n")
                
                # Parse
                t1 = TimestampParser.extract(outputs[0]) if len(outputs) > 0 else []
                t2 = TimestampParser.extract(outputs[1]) if len(outputs) > 1 else []
                local_timestamps = TimestampParser.merge_and_sort(t1, t2)
            else:
                local_timestamps = []

        # Map to global timestamps
        global_timestamps = [map_timestamp_to_global(ts, start_sec) for ts in local_timestamps]
        all_moments[f"Round {round_num}"] = global_timestamps

    if log_cb: log_cb(f"\nCompleted finding moments: {all_moments}")
    return all_moments
