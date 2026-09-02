# Boxing Round Extractor Logic

This document explains the core logic mechanisms used to extract boxing rounds from chaotic TV broadcasts using OCR and Audio Verification.

## 1. Handling OCR Hallucinations

Raw OCR often produces garbage text or misinterprets fonts (e.g., misreading "11" as "1I"). The engine defends against this using three layers:

- **Strict Regex:** Word boundaries (`\b`) ensure attached characters (like `1I`) fail the match. Impossible formats (e.g., `Round 12 of 8`) are mathematically rejected.

- **Cluster Consensus (Size > 1):** Every clock reading is projected into absolute start/end times (`sec - (rl - clk)`). These projections form clusters. Any cluster with only 1 vote is instantly deleted, destroying one-off OCR glitches.

- **Start/End Anchoring:** If a hallucination is thick enough to survive the size check (e.g., 10 bad reads for Round 1 occurring 40 minutes into the fight), it is caught by anchoring. The end-time cluster of a round is strictly anchored to its start-time cluster. Any end cluster that falls outside a physically possible 10-minute window from the start is rejected.

## 2. Handling Pauses

Pauses stop the broadcast clock while absolute video time keeps ticking, causing mathematical projections to shift backwards. 

- **Pre/Post-Pause Clusters:** A round with a pause naturally generates two separate clusters of projected start times: a "pre-pause" cluster (later absolute time) and a "post-pause" cluster (earlier absolute time).

- **Unassigned Merging:** If the broadcast hides the round number during a pause, the orphaned clock readings are dynamically merged back into the active round if they fall within 60 seconds of the core cluster.

- **Min/Max Bounding:** To determine the true bounds of the round, the script takes `min(valid_clusters)` to find the true START time (before the pause) and `max(valid_clusters)` to find the true END time (after the pause extensions).

## 3. Handling Knockouts (KOs)

When a KO happens early in a round, the clock is permanently removed. Standard extrapolation would assume the round lasted the full 3 minutes.

- **KO Discrepancy Check:** The script tracks `max_sec` (the absolute latest video timestamp the clock was visible). For the final round of the fight, it compares the projected 3-minute end time against this `max_sec`.

- **Truncation:** If the clock vanished more than 35 seconds before the round was projected to end, the engine flags it as a KO. It instantly overwrites the 3-minute end time with `max_sec + pad` (padding it slightly to capture the referee wave-off).

## 4. Handling Missing Round Numbers
Broadcasts occasionally show the clock but completely forget to render the text "Round 1". 

- **Unassigned Pools:** Clock readings without round numbers are dumped into an `unassigned_starts` pool and clustered.

- **Temporal Extrapolation:** After building the schedule for all detected rounds (e.g., Rounds 2, 3, 4), the script realizes Round 1 is missing. It mathematically extrapolates where Round 1 should be (Round 2 start minus ~240s).

- **Cluster Snapping:** Instead of blindly guessing, the script scans the unassigned pool and snaps Round 1 to the nearest valid, orphaned cluster that happened before Round 2. 

## 5. Sampling Density (250 vs 500)

More samples are not always better. 

At **500 samples** (every ~7s), graphical glitches that stay on screen for 20 seconds will be scanned 3 times, forming a cluster of 3, which tricks the `size > 1` consensus check into accepting the hallucination.

At **250 samples** (every ~14s), that same glitch is only scanned once. It forms a cluster of 1, which fails the consensus check and is instantly deleted. 250 samples acts as a natural low-pass filter, starving hallucinations of the votes they need to survive while true clock readings (on screen for 180s) easily survive.


## Quick Start

```sh
cd Round_extract_web/
uvicorn app.server:app --reload  # backend
```

```sh
cd Round_extract_web/frontend
python -m http.server 8080  # frontend
```

Open `http://localhost:8080` in your browser