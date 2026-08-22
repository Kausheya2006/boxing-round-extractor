import subprocess
from app.__init__ import *

class RoundExtractor:

    def __init__(self, video_path, time_between_rounds_sec=240, round_length = None, total_rounds=None):
        self.video_path = video_path
        self.time_between_rounds_sec = time_between_rounds_sec
        self.total_rounds = total_rounds
        self.round_length = round_length

        if os.getenv("MOCK_MODE") == "True":
            self.ocr = None  
        else:
            self.ocr = PaddleOCR(use_angle_cls=False, lang='en')

    def preprocess_frame(self, frame):
        """Enhances the frame for better OCR accuracy."""
        # Keep original resolution but crop out the middle 50% of the screen.
        # Boxing UI is almost always in the top 25% or bottom 25%.
        h, w = frame.shape[:2]
        top_crop = frame[0:int(h*0.25), :]
        bottom_crop = frame[int(h*0.75):h, :]
        frame = cv2.vconcat([top_crop, bottom_crop])
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.5, beta=0)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _parse_ocr(self, result):
        """Extracts clock, round, and total rounds from OCR text block."""
        if not result or not result[0]: 
            return None, None, None, ""
        
        # Flatten all detected text into one string for faster Regex searching
        text = " ".join([t.strip().upper() for t in result[0].get('rec_text', result[0].get('rec_texts', []))])
        
        t_match = re.search(r'([0-3]):([0-5][0-9])', text)
        # Match 'RND 3 OF 12', '3 OF 12', '3 | 12', '3 12', or fallback to 'RND 3'
        r_match = re.search(r'(?:RND|ROUND)?\s*(\d+)\s*(?:OF|\||/|-|\s+)\s*(4|6|8|10|12)\b|(?:RND|ROUND)\s*(\d+)', text)
        
        clock = int(t_match[1]) * 60 + int(t_match[2]) if t_match else None
        rnd = int(r_match[1] or r_match[3]) if r_match else None
        tot = int(r_match[2]) if r_match and r_match.lastindex and r_match.lastindex >= 2 and r_match[2] else None
        
        return clock, rnd, tot, text

    def _get_consensus(self, values, mode="start", tol=3.0):
        """
        Groups similar timestamp guesses into clusters. 
        Returns the min (if start) or max (if end) of valid clusters (len > 1).
        """
        if not values:
            return None
            
        clusters = []
        for v in sorted(values):
            if not clusters or v - clusters[-1][-1] > tol:
                clusters.append([v])
            else:
                clusters[-1].append(v)
                
        valid_clusters = [c for c in clusters if len(c) > 1]
        
        # Fallback if we don't have ANY cluster > 1
        if not valid_clusters:
            valid_clusters = clusters
            
        if mode == "start":
            best_cluster = min(valid_clusters, key=lambda c: c[0])
            return min(best_cluster)
        else:
            best_cluster = max(valid_clusters, key=lambda c: c[0])
            return max(best_cluster)

    def get_schedule(self, num_samples=150, reuse_debug_log=False, log_cb=None, check_cancel=None):
        """Main orchestrated loop for sampling, parsing, and clustering."""
        
        if log_cb: log_cb(f"Starting uniform sampling of {num_samples} frames...")
        
        if not self.ocr: 
            if not self.total_rounds:
                self.total_rounds = 2
            if not self.round_length:
                self.round_length = 180
            return {1: 60.0, 2: 300.0} # Mock Mode

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): 
            return None

        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS)) if cap else 3600
        reads, max_clk = [], 0
        
        video_stem = os.path.splitext(os.path.basename(self.video_path))[0]
        debug_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", f"debug_ocr_{video_stem}.txt")
        
        if reuse_debug_log and os.path.exists(debug_file_path):
            if log_cb: log_cb("Fast Debug mode: Parsing existing debug_ocr.txt...")
            with open(debug_file_path, "r") as f:
                for line in f:
                    match = re.search(r'at ([\d\.]+)s .* Parsed: Clk=(\d+|None), Rnd=(\d+|None), Tot=(\d+|None)', line)
                    if match:
                        sec = float(match.group(1))
                        clk = int(match.group(2)) if match.group(2) != "None" else None
                        rnd = int(match.group(3)) if match.group(3) != "None" else None
                        tot = int(match.group(4)) if match.group(4) != "None" else None
                        
                        if clk is not None:
                            max_clk = max(max_clk, clk)
                            if rnd:
                                reads.append((rnd, sec, clk, tot))
            cap.release()
            # Approximate dur if cap was bad, otherwise keep dur
            if not dur or dur <= 0:
                dur = max([s for r, s, c, t in reads]) + 300 if reads else 3600
        else:
            if log_cb: log_cb("Testing OCR engine on the first frame...")
            ret, frame = cap.read()
            if ret:
                try:
                    processed_frame = self.preprocess_frame(frame)
                    self.ocr.predict(processed_frame, use_doc_orientation_classify=False, use_doc_unwarping=False)
                    if log_cb: log_cb("OCR is working.")
                except Exception as e:
                    if log_cb: log_cb(f"Warning: OCR test failed on first frame: {e}")

            with open(debug_file_path, "w") as f:
                f.write(f"--- OCR Debug Log for {self.video_path} ---\n")

            # 1. Sweep the video uniformly

            samples = [dur * (i / (num_samples + 1)) for i in range(1, num_samples + 1)]  # uniformly spaced timestamps

            for i, sec in enumerate(samples):
                if check_cancel and check_cancel():
                    if log_cb: log_cb("OCR extraction cancelled by user.")
                    raise Exception("Cancelled by user")

                if log_cb and i % 5 == 0: 
                    log_cb(f"Processing sample {i+1}/{num_samples}...")

                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # offset
                ret, frame = cap.read()

                if not ret: 
                    continue

                processed_frame = self.preprocess_frame(frame)

                res = self.ocr.predict(processed_frame, use_doc_orientation_classify=False, use_doc_unwarping=False)
                
                clk, rnd, tot, raw_text = self._parse_ocr(res) # clock-time, round-number, total-rounds

                with open(debug_file_path, "a") as f:
                    f.write(f"Sample {i+1} at {sec:.1f}s | Text: {raw_text} | Parsed: Clk={clk}, Rnd={rnd}, Tot={tot}\n")

                if clk is not None:
                    max_clk = max(max_clk, clk)  # needed for dynamic round length (2 or 3 minutes?)
                    if rnd: 
                        reads.append((rnd, sec, clk, tot))
                    
            cap.release()

        # 2. Dynamic format detection (3-min vs 2-min) and consensus math
        
        rl = self.round_length
        if rl is None:
            rl = 180 if max_clk > 120 else 120 
            self.round_length = rl
            if log_cb: log_cb(f"Dynamic round length determined: {rl} seconds")

        # 2. Back-calculate and Forward-calculate true start & end times

        start_votes = {}
        end_votes = {}
        
        for rnd, sec, clk, tot in reads:
            start_sec = sec - (rl - clk)  # back-calculate the start of the round
            end_sec = sec + clk           # forward-calculate the end of the round
            
            if 0 <= start_sec <= dur:
                start_votes.setdefault(rnd, []).append(start_sec) 
            if 0 <= end_sec <= dur:
                end_votes.setdefault(rnd, []).append(end_sec)
                
            if tot is not None:
                if not self.total_rounds or tot > self.total_rounds:
                    self.total_rounds = tot

        schedule = {}
        for r in sorted(start_votes.keys()):
            # Log the raw votes so we can see the clusters
            if log_cb: 
                log_cb(f"Round {r} Start Votes: {[round(v, 1) for v in sorted(start_votes.get(r, []))]}")
                log_cb(f"Round {r} End Votes: {[round(v, 1) for v in sorted(end_votes.get(r, []))]}")
                
            start_val = self._get_consensus(start_votes.get(r, []), mode="start")
            end_val = self._get_consensus(end_votes.get(r, []), mode="end")
            if start_val is not None and end_val is not None:
                schedule[r] = (start_val, end_val)
                
        return schedule

    def run(self, num_samples=150, reuse_debug_log=False, log_cb=None, check_cancel=None):
        """Bridges gaps and outputs the final MM:SS strings."""
        if log_cb: log_cb("Extracting round start times via OCR...")
        sched = self.get_schedule(num_samples=num_samples, reuse_debug_log=reuse_debug_log, log_cb=log_cb, check_cancel=check_cancel)

        if not sched: 
            if log_cb: log_cb("Failed to extract schedule.")
            return None
            
        if not self.total_rounds:
            self.total_rounds = max(sched.keys()) if sched else 12

        # Anchor Round 1 if missing
        earliest = min(sched.keys())
        if 1 not in sched:
            # Extrapolate backwards: start = earliest_start - (r-1)*(rl+60)
            diff = (earliest - 1) * self.time_between_rounds_sec
            start_1 = max(0, sched[earliest][0] - diff)
            sched[1] = (start_1, start_1 + self.round_length)

        final_ts = []
        from app.utils import sec_to_time

        max_detected_round = max(sched.keys())

        for r in range(1, max_detected_round + 1):
            if r not in sched:
                # Extrapolate forwards from r-1: previous end + 60s break
                prev_end = sched[r-1][1]
                start_r = prev_end + 60
                sched[r] = (start_r, start_r + self.round_length)
                
            start_sec, end_sec = sched[r]
            final_ts.append(sec_to_time(start_sec))
            final_ts.append(sec_to_time(end_sec))
            
        if log_cb: log_cb(f"Extracted {len(final_ts)//2} rounds.")

        return final_ts, self.round_length, self.total_rounds

