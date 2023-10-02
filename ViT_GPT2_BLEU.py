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


import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

class config : 
    ENCODER = "google/vit-large-patch16-224" # "google/vit-base-patch16-224" 
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
    
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

feature_extractor = ViTImageProcessor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token    

transforms = transforms.Compose(
    [
        transforms.Resize(config.IMG_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.5, 
            std=0.5
        )
   ]
)
df = pd.read_csv('results.csv', sep='|', index_col=False)
print("Total Results DataFrame Samples = ", len(df))
df = df.drop(columns=[' comment_number'], axis=1)
df = df.rename(columns={'image_name':'image', ' comment':'caption'})

train_df , val_df = train_test_split(df , test_size = 0.1 ,shuffle = False, stratify = None)
print(train_df)
print(val_df)
print("Total Training Samples = ", len(train_df),"\t Total Validation Samples = ", len(val_df))


class ImgDataset(Dataset):
    def __init__(self, df,root_dir,tokenizer,feature_extractor, transform = None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer= tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 50
    def __len__(self,):
        return len(self.df)
    def __getitem__(self,idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir , image)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img= self.transform(img)
        
        # Normalize pixel values to [0, 1]
        img = (img - img.min()) / (img.max() - img.min())
    
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        captions = self.tokenizer(str(caption),
                                 padding='max_length',
                                 max_length=self.max_length,
                                 truncation=True,
                                 return_tensors='pt'
                                 ).input_ids

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": captions.clone().detach()}
        # torch.tensor(captions) = captions.clone().detach()
        return encoding

        
        
train_dataset = ImgDataset(train_df, root_dir = 'flickr30k_images',tokenizer=tokenizer,feature_extractor = feature_extractor ,transform = transforms)
val_dataset = ImgDataset(val_df , root_dir = 'flickr30k_images',tokenizer=tokenizer,feature_extractor = feature_extractor , transform  = transforms)  



# Define the paths to your model files
file_path = 'VIT_large_gpt2_epoch_3.0' # CHOOSE THE CORRECT MODEL
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


def calculate_bleu(references, hypotheses):
    # Calculate BLEU-1,2,3,4 score for a single reference and hypothesis
    print(corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0)))
    print(corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0)))
    print(corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)))
    print(corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)))

# Example usage
references = []
hypotheses = []

ctr = 1
for item in val_dataset:
    print("-"*60,"ITERATION ", ctr, "-"*60)
    ctr += 1
    references.append([tokenizer.decode(item["labels"][0].tolist(), skip_special_tokens=True)])
    input_ids = item["pixel_values"].unsqueeze(0).to(device)
    generated_ids = model.generate(input_ids)
    hypotheses.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

calculate_bleu(references, hypotheses)
