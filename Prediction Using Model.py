#!/usr/bin/env python
# coding: utf-8

# In[6]:


import transformers
print(transformers.__version__)


# In[3]:


# Importing the required packages
import transformers
import PIL
import requests
import torchvision
import torch

# Getting the version of each package
transformers_version = transformers.__version__
PIL_version = PIL.__version__
requests_version = requests.__version__
torchvision_version = torchvision.__version__
torch_version = torch.__version__

transformers_version, PIL_version, requests_version, torchvision_version, torch_version


# In[2]:


from transformers import VisionEncoderDecoderModel, AutoTokenizer
from PIL import Image
import requests
import torchvision.transforms as T
import torch

# Load the tokenizer and model
model_name = "ViT_Huge_and_GPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)


# In[16]:


image_path = '//Users/abhishekkarmakar/Desktop/ViT/teacher.jpeg'
# image_path = "two dogs.png"
image = Image.open(image_path).convert('RGB') 

transform = T.Compose([
    T.Resize((224, 224)),  
    T.ToTensor(),          
    T.Normalize(0.5, 0.5) 
])

# Apply the transformation to the image
transformed_image = transform(image).unsqueeze(0)  # Add a batch dimension

# Generate text
output_ids = model.generate(transformed_image)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

generated_text = generated_text[0].upper() + generated_text[1:]
print(generated_text)

