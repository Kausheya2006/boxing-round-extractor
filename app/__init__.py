import cv2
import re
import logging
import numpy as np
import os
import subprocess

import torch
import librosa
from transformers import ClapModel, ClapProcessor



os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEMP_AUDIO_PATH = "../assets/temp/"