from app.__init__ import *

_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        import torch
        torch.set_num_threads(4)
        _ocr_reader = easyocr.Reader(['en'], gpu=True, quantize=True)
    return _ocr_reader

class RoundExtractor:

    def __init__(self, video_path, time_between_rounds_sec=240, round_length=None, total_rounds=None):
        self.video_path = video_path
        self.time_between_rounds_sec = time_between_rounds_sec
        self.total_rounds = total_rounds
        self.round_length = round_length

        if os.getenv("MOCK_MODE") == "True":
            self.ocr = None  
        else:
            self.ocr = get_ocr_reader()

    def preprocess_frame(self, frame):
        """4-corner crop + light vertical dilation for ':' and '|' retention."""
        h, w = frame.shape[:2]
        y_pct, x_pct = int(h * 0.20), int(w * 0.30)
        
        top_left = frame[0:y_pct, 0:x_pct]
        top_right = frame[0:y_pct, w-x_pct:w]
        bottom_left = frame[h-y_pct:h, 0:x_pct]
        bottom_right = frame[h-y_pct:h, w-x_pct:w]
        
        top_row = cv2.hconcat([top_left, top_right])
        bottom_row = cv2.hconcat([bottom_left, bottom_right])
        combined = cv2.vconcat([top_row, bottom_row])
        
        # 1.2x scale keeps speed high while avoiding glyph blurring
        resized = cv2.resize(combined, (0, 0), fx=1.1, fy=1.1, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Vertical dilation prevents colon dots and divider lines from disappearing
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        thickened = cv2.morphologyEx(gray, cv2.MORPH_DILATE, vert_kernel)
        
        return cv2.cvtColor(thickened, cv2.COLOR_GRAY2RGB)
    
    def _parse_ocr(self, result):
        """Extracts clock, round, and total rounds from OCR text block."""
        if not result: 
            return None, None, None, ""
        
        # Flatten all detected text into one string for faster Regex searching
        text = " ".join([t.strip().upper() for t in result])
        
        # New clock matching handles '2:31', '2.31', '1;08', or '1837' (colon read as 8)
        t_match = re.search(r'\b([0-3]?\d)[:.;8]([0-5]\d)\b', text)
        clock = int(t_match.group(1)) * 60 + int(t_match.group(2)) if t_match else None
        
        # Match 'RND 3 OF 12', '3 OF 12', '3 | 12', '3 12', or fallback to 'RND 3'
        r_matches = list(re.finditer(r'(?:RND|ROUND)?\s*(\d+)\s*(?:OF|\||/|-|\s+)\s*(4|6|8|10|12)\b|(?:RND|ROUND)\s*(\d+)\b', text))
        
        rnd = None
        tot = None
        
        # EasyOCR often puts garbage on the left side of the screen (e.g. "1 4"). 
        # By iterating backwards and taking the last valid match, we usually hit the true UI text in the center/right.
        for r_match in reversed(r_matches):
            possible_rnd = int(r_match.group(1) or r_match.group(3))
            possible_tot = int(r_match.group(2)) if r_match.lastindex and r_match.lastindex >= 2 and r_match.group(2) else None
            
            # Guard against absurd numbers (e.g. 282) or zeros
            if 0 < possible_rnd <= 20:
                # Guard against backwards parsing (e.g. '12 8' from 'ROUND OF 12 8' where '4' was dropped)
                if possible_tot and possible_rnd > possible_tot:
                    continue
                    
                rnd = possible_rnd
                tot = possible_tot
                break
        
        return clock, rnd, tot, text

    def _get_consensus(self, votes, mode="start", anchor=None):
        if not votes: return None
        clusters = []
        for v in sorted(votes):
            if not clusters or v - clusters[-1][-1] > 10.0:
                clusters.append([v])
            else:
                clusters[-1].append(v)
        
        # Must have at least 2 votes to prevent a random hallucination
        valid_clusters = [c for c in clusters if len(c) > 1]
        if not valid_clusters:
            valid_clusters = clusters # fallback
            
        if anchor is not None:
            # Filter clusters to only those that make physical sense relative to the anchor
            if mode == "end":
                valid_clusters = [c for c in valid_clusters if 0 < max(c) - anchor < 600]
            elif mode == "start":
                valid_clusters = [c for c in valid_clusters if 0 < anchor - min(c) < 600]
                
        if not valid_clusters:
            return None
            
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
            return {1: 60.0, 2: 300.0}, [] # Mock Mode

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
                            # Append even if rnd is missing so we can use unlabelled clocks!
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
                    import torch
                    with torch.no_grad():
                        self.ocr.readtext(processed_frame, detail=0, canvas_size=640, mag_ratio=1.0)
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

                import torch
                with torch.no_grad():
                    res = self.ocr.readtext(processed_frame, detail=0, canvas_size=640, mag_ratio=1.0)
                
                clk, rnd, tot, raw_text = self._parse_ocr(res) # clock-time, round-number, total-rounds

                with open(debug_file_path, "a") as f:
                    f.write(f"Sample {i+1} at {sec:.1f}s | Text: {raw_text} | Parsed: Clk={clk}, Rnd={rnd}, Tot={tot}\n")

                if clk is not None:
                    max_clk = max(max_clk, clk)  # needed for dynamic round length (2 or 3 minutes?)
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
        unassigned_starts = []
        unassigned_ends = []
        
        for rnd, sec, clk, tot in reads:
            start_sec = sec - (rl - clk)  # back-calculate the start of the round
            end_sec = sec + clk           # forward-calculate the end of the round
            
            if 0 <= start_sec <= dur:
                if rnd:
                    start_votes.setdefault(rnd, []).append((start_sec, sec)) 
                else:
                    unassigned_starts.append((start_sec, sec))
            if 0 <= end_sec <= dur:
                if rnd:
                    end_votes.setdefault(rnd, []).append((end_sec, sec))
                else:
                    unassigned_ends.append((end_sec, sec))
                
            if tot is not None:
                if not self.total_rounds or tot > self.total_rounds:
                    self.total_rounds = tot

        # Merge orphaned clocks that logically belong to detected rounds
        start_anchors = {r: sorted([v[0] for v in votes])[len(votes)//2] for r, votes in start_votes.items()}
        remaining_unassigned_starts = []
        for u_start, u_sec in unassigned_starts:
            if not start_anchors:
                remaining_unassigned_starts.append((u_start, u_sec))
                continue
            closest_r = min(start_anchors.keys(), key=lambda r: abs(start_anchors[r] - u_start))
            if abs(start_anchors[closest_r] - u_start) < 60:
                start_votes[closest_r].append((u_start, u_sec))
            else:
                remaining_unassigned_starts.append((u_start, u_sec))
                
        end_anchors = {r: sorted([v[0] for v in votes])[len(votes)//2] for r, votes in end_votes.items()}
        for u_end, u_sec in unassigned_ends:
            if not end_anchors: continue
            closest_r = min(end_anchors.keys(), key=lambda r: abs(end_anchors[r] - u_end))
            if abs(end_anchors[closest_r] - u_end) < 60:
                end_votes[closest_r].append((u_end, u_sec))

        schedule = {}
        max_r = max(start_votes.keys()) if start_votes else 0
        for r in sorted(start_votes.keys()):
            s_votes_full = start_votes.get(r, [])
            e_votes_full = end_votes.get(r, [])
            
            start_val = self._get_consensus([v[0] for v in s_votes_full], mode="start")
            end_val = self._get_consensus([v[0] for v in e_votes_full], mode="end", anchor=start_val)
            
            if start_val is not None and end_val is not None:
                # KO Detection: If this is the last detected round and the clock permanently disappeared 
                # long before the projected end, the round ended early (KO).
                if r == max_r and e_votes_full:
                    max_sec = max([v[1] for v in e_votes_full])
                    
                    # Also apply padding using the sampling interval logic the user approved!
                    # If dur/num_samples = 7.2s, we pad it to make sure we don't cut right before the wave off
                    pad = (dur / num_samples) + 5
                    
                    if end_val - max_sec > 35:
                        end_val = max_sec + pad
                        
                schedule[r] = (start_val, end_val)
                
        # Cluster unassigned starts (clocks with missing round numbers) for missing rounds
        unassigned_clusters = []
        for v, _ in sorted(remaining_unassigned_starts, key=lambda x: x[0]):
            if not unassigned_clusters or v - unassigned_clusters[-1][-1] > 5.0:
                unassigned_clusters.append([v])
            else:
                unassigned_clusters[-1].append(v)
        valid_unassigned_starts = [min(c) for c in unassigned_clusters if len(c) > 1]
                
        return schedule, valid_unassigned_starts

    def run(self, num_samples=150, reuse_debug_log=False, log_cb=None, check_cancel=None):
        """Bridges gaps and outputs the final MM:SS strings."""
        if log_cb: log_cb("Extracting round start times via OCR...")
        res = self.get_schedule(num_samples=num_samples, reuse_debug_log=reuse_debug_log, log_cb=log_cb, check_cancel=check_cancel)

        if not res: 
            if log_cb: log_cb("Failed to extract schedule.")
            return None
            
        sched, unassigned_starts = res
        
        if not sched:
            if log_cb: log_cb("Failed to extract schedule.")
            return None
            
        if not self.total_rounds:
            self.total_rounds = max(sched.keys()) if sched else 12

        # Anchor Round 1 if missing
        earliest = min(sched.keys())
        if 1 not in sched:
            diff = (earliest - 1) * self.time_between_rounds_sec
            expected_start = max(0, sched[earliest][0] - diff)
            
            # Find an unlabelled clock cluster that is before Round 2
            actual_start = expected_start
            candidates = [u for u in unassigned_starts if u < sched[earliest][0]]
            if candidates:
                # Pick the unassigned start closest to our expected start
                actual_start = min(candidates, key=lambda x: abs(x - expected_start))
                
            sched[1] = (actual_start, actual_start + self.round_length)

        final_ts = []
        from app.utils import sec_to_time

        max_detected_round = max(sched.keys())

        for r in range(1, max_detected_round + 1):
            if r not in sched:
                # Extrapolate forwards from r-1: previous end + 60s break
                prev_end = sched[r-1][1]
                expected_start = prev_end + 60
                
                # Check for an unlabelled clock cluster that fits here
                actual_start = expected_start
                candidates = [u for u in unassigned_starts if u > prev_end]
                # Must be before the NEXT known round start, if it exists
                next_rounds = [sched[nxt][0] for nxt in sched.keys() if nxt > r]
                if next_rounds:
                    candidates = [u for u in candidates if u < min(next_rounds)]
                    
                if candidates:
                    actual_start = min(candidates, key=lambda x: abs(x - expected_start))
                    
                sched[r] = (actual_start, actual_start + self.round_length)
                
            start_sec, end_sec = sched[r]
            final_ts.append(sec_to_time(start_sec))
            final_ts.append(sec_to_time(end_sec))
            
        if log_cb: log_cb(f"Extracted {len(final_ts)//2} rounds.")

        return final_ts, self.round_length, self.total_rounds

