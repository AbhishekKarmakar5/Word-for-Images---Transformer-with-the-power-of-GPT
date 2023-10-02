
import os
import warnings
warnings.filterwarnings("ignore")

import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel , VisionEncoderDecoderConfig, ViTImageProcessor
from transformers import AutoTokenizer ,  GPT2Config # default_data_collator
from transformers import TrainerCallback 

def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

class config : 
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    NUM_WORKERS = mp.cpu_count()
    EPOCHS = 10
    IMG_SIZE = (224,224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95
    
feature_extractor = ViTImageProcessor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token   

# Define the paths to your model files
file_path = 'VIT_large_gpt2_epoch_2.0'
model_path = file_path + '/pytorch_model.bin'
config_path = file_path + '/config.json'

# Load the model's state dictionary
state_dict = torch.load(model_path)

# Load the model's configuration
config = VisionEncoderDecoderConfig.from_json_file(config_path)

# Initialize the model
model = VisionEncoderDecoderModel(config=config)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval() 

# Move the model to the desired device (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

img =  Image.open("flickr30k_images/10287332.jpg").convert("RGB")
#img =  Image.open("flickr30k_images/101262930.jpg").convert("RGB")
#img =  Image.open("flickr30k_images/1000092795.jpg").convert("RGB")

generated_caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cuda"))[0], skip_special_tokens=True)
print(generated_caption)