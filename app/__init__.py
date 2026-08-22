import cv2
import re
from paddleocr import PaddleOCR

import logging
import numpy as np
import os
import subprocess

import torch
import librosa
from transformers import ClapModel, ClapProcessor

logging.getLogger("ppocr").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEMP_AUDIO_PATH = "../assets/temp/"