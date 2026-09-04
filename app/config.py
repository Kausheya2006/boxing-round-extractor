from app.__init__ import *

AUDIO_MODEL = "laion/clap-htsat-unfused"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

GEMINI_MODEL = "models/gemini-3.5-flash"

MOMENTS_PROMPT = """
Identify key moments in this video using both video action and commentator audio cues (e.g., sudden yelling, "Down goes...", "He's hurt!", or "It's over!").

Flag an event ONLY if it meets one of these:
1. Knockdown (canvas contact, referee count, or commentator calls a knockdown)
2. Stoppage (KO/TKO, referee waves off fight, or fight is declared over)
3. Fighter Severely Hurt (visible stumble/wobble confirmed by excited commentary)
4. Pinned/Corner Trap (fighter trapped against ropes/corner absorbing a flurry)

Output format:
- [MM:SS] Event: [Knockdown / Stoppage / Hurt / Pinned] | Reason: [Visual proof + commentator cue]

If none occurred, reply: "No key moments detected."
"""

VERIFY_MOMENTS_MODEL = "models/gemini-3.1-pro-preview"

VERIFY_MOMENTS_PROMPT = """
You must output all three numbered sections below. Do not skip directly to the final list.

### 1. Visual Evidence (Write out each point):
- Attacker & Defender: [Identify by shorts/torso color]
- Action: [Punch thrown or action taken]
- Contact: [Clean hit / Hit guard / Complete miss]
- Defender Base & Footing: [Did feet move? Did knees buckle, or was footing solid?]
- Referee Action: [No action / Intervening / Counting / Waving off]

### 2. Verdict:
- Is this an unambiguous Key Moment (Knockdown, Stoppage, Stagger/Hurt, or Pinned Flurry)?: [YES / NO]

### 3. Structured Data:
Output this exact line at the very end:
RESULT: [YES/NO, "Category", "Landed_Status", "Attacker vs Defender", "Attack_Type", "Defender_Reaction"]
(If NO, output: RESULT: [NO, null, null, null, null, null])
"""