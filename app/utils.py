def time_to_sec(time_str):
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

def sec_to_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


candidate_labels = [
    "a loud, sharp, metallic boxing match bell ringing",  # target
    
    "a sports commentator speaking excitedly into a microphone", 
    
    "a massive stadium crowd cheering and clapping loudly", 
    
    "quiet ambient room tone or silence"
]
