# Word-for-Images - Transformer with the Power of GPT

This project combines the power of Vision Transformer (ViT) as an encoder for feature extraction with GPT2 as the decoder for generating text or captions for images. The goal is to evaluate the quality of generated text using the BLEU (Bilingual Evaluation Understudy) score.

## BLEU Scores

Here are the BLEU scores obtained using ViT Base + GPT2:

- BLEU 1: 0.60
- BLEU 2: 0.45
- BLEU 3: 0.34
- BLEU 4: 0.22

## Datasets

We used the following standard datasets for testing and training purposes:

- Flickr8K
- Flickr30k

## Project Details

For more detailed information about the project, you can refer to the following resources:

- [Project Report](https://docs.google.com/document/d/1KC35BhD_usZiRt6Okynyiw53U2OnsEEe-Nc2xlnglmo/edit?pli=1#heading=h.ajwnem8pvzkh)
- [Project Presentation](https://docs.google.com/presentation/d/1Ue2c5Xqicji7wRZ4VhuBz4y3nRs6WKcdqDeoU--HqIk/edit?pli=1#slide=id.g2580ba7db4c_1_0)

## Image Caption Generation Alternative

Additionally, another implementation of image caption generation is provided using the following method:

- Feature Extractor: VGG19
- Text Generator: Dense LSTM

You can find the code for this alternative implementation in the GitHub repository [here](https://github.com/Ab
