from app.__init__ import *

model_id = "laion/clap-htsat-unfused"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"