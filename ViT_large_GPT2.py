import os

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
from transformers import VisionEncoderDecoderModel , ViTImageProcessor
from transformers import AutoTokenizer ,  GPT2Config # default_data_collator
from transformers import TrainerCallback

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
class config : 
    ENCODER = "google/vit-large-patch16-224" # "google/vit-base-patch16-224" # "google/vit-huge-patch14-224-in21k"
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
    EPOCHS = 3
    IMG_SIZE = (224,224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95
    
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

sacrebleu = datasets.load_metric("sacrebleu")
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    
    pred_str = [pred_str]  # Convert the string to a list containing the string
    label_str = [label_str]  # Convert the string to a list containing the string

    # https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization.ipynb#scrollTo=5o4rUteaIrI_
    results = sacrebleu.compute(predictions=pred_str, references=label_str)

    return {
        "Bleu_score": round(results["score"],4),
        "Precisions": results["precisions"], # geometric mean of n-gram precisions
        "Brevity_Penalty": round(results["bp"],4),
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }  
    
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

print(df.head())
print("-"*120)

train_df , val_df = train_test_split(df , test_size = 0.1 ,shuffle = False, stratify = None)
print(train_df)
#train_df.to_csv("train_df.csv")
print("-"*120)

print(val_df)
#val_df.to_csv("val_df.csv")
print("-"*120)
#-------------------------------------------------------------------------------------------
# train_df = train_df.head(10)
# val_df = val_df.head(10)
#-------------------------------------------------------------------------------------------

print("Total Training Samples = ", len(train_df),"\t Total Validation Samples = ", len(val_df)) # 28604 train + 3,178 test images
del df

class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Save the model after each epoch
        model.save_pretrained(f"VIT_large_gpt2_epoch_{state.epoch}")


class ImgDataset(Dataset):
    def __init__(self, df,root_dir,tokenizer,feature_extractor, transform = None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer= tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 75 # 50 <- Picked from VGG19 + Dense LSTM 
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
        #captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]

        
        """
        # Check if pixel_values and captions have the same size along the first dimension
        if pixel_values.size(0) != len(captions):
            # Adjust the size of pixel_values to match the length of captions
            pixel_values = pixel_values[:len(captions)]
        """
            
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": captions.clone().detach()}
        # torch.tensor(captions) = captions.clone().detach()
        return encoding

        
        
train_dataset = ImgDataset(train_df, root_dir = 'flickr30k_images',tokenizer=tokenizer,feature_extractor = feature_extractor ,transform = transforms)
val_dataset = ImgDataset(val_df , root_dir = 'flickr30k_images',tokenizer=tokenizer,feature_extractor = feature_extractor , transform  = transforms)  

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
# set beam search parameters
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


training_args = Seq2SeqTrainingArguments(
    output_dir='VIT_large_gpt2',
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,  
    save_steps=2048, 
    warmup_steps=1024,  
    learning_rate = 5e-5,
    #max_steps=1500, # delete for full training
    num_train_epochs = config.EPOCHS, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1,
)

def default_data_collator(features):
    batch = {k: torch.stack([f[k].squeeze() for f in features]) for k in features[0]}
    return batch
    
# instantiate trainer
trainer = Seq2SeqTrainer(
    #tokenizer=feature_extractor,
    tokenizer = tokenizer,
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)

# Instantiate the SaveModelCallback - save the model after every epoch
save_model_callback = SaveModelCallback()
# Add the callback to the list of callbacks in the Trainer
trainer.add_callback(save_model_callback)


trainer.train()
trainer.save_model('VIT_large_gpt2')
