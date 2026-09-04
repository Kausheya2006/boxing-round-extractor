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
Analyze this 2-second clip and answer each step in order:

1. Physical Reaction: Did the defending fighter drop to the canvas, buckle at the knees, slip/trip, take involuntary stumbling steps, or get pinned against ropes? [YES / NO]
2. Is this a real Key Moment or Official Event? (Knockdown, Accidental Slip to canvas, Stoppage, Stagger/Hurt, or Pinned Flurry): [YES / NO]
If NO, state why briefly and output at the end: RESULT: [NO, null, null, null, null, null]

If YES, complete the remaining checks:
3. Category: [Knockdown Count / Accidental Slip / Referee Stoppage / Staggering Power Shot / Corner Pin Flurry]
4. Contact: Did the attack land cleanly?: [Clean Hit / Guard Blocked / Missed / No Punch (Trip/Push)]
5. Who hit whom?: [Attacker shorts/torso color] vs [Defender shorts/torso color]
6. Attack Type: [e.g., Left Hook, Straight Right/Cross, Uppercut, Flurry, None/Push]
7. Defender Reaction: [Fell to Canvas, Slipped/Tripped, Knees Buckled, Stumbled Backward, Pinned Against Ropes]

Output your brief observations first, then end strictly with this line:
RESULT: [YES/NO, "Category", "Landed_Status", "Attacker vs Defender", "Attack_Type", "Defender_Reaction"]
"""