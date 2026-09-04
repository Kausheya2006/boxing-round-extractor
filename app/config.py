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
Analyze this 2-second clip frame-by-frame. Work through these observation steps before giving a verdict:

1. Frame Breakdown:
- Attacker & Defender: Identify fighters by torso/shorts appearance.
- Action: What punch or motion is thrown? (e.g., Left Hook, Straight Right, Clinch, None)
- Contact: Did it land cleanly, hit the guard, or miss?
- Defender Reaction: Did the defender's base stay firm, or did they fall, buckle knees, stumble, or get pinned against ropes?
- Referee: Did the referee intervene, begin a count, or wave off the fight?

2. Key Moment Assessment:
- Based on the reaction above, does this qualify as a true Key Moment (Knockdown, Stoppage, Stagger/Hurt, or Pinned Flurry)?: [YES / NO]

3. Final Output:
If NO: [NO, null, null, null, null, null]
If YES: [YES, "Category (2-3 words)", Landed_Status, "Attacker vs Defender", "Attack_Type", "Defender_Reaction"]
"""