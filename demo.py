from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoConfig
import torch.nn as nn
from tqdm import tqdm
from eva_clip import create_model_and_transforms, create_model_from_pretrained
from training.llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
import torch.nn.functional as F
import argparse
import torch
import logging
from PIL import Image
from eva_clip import create_model_and_transforms, create_model_from_pretrained
from training.llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
import torch.nn as nn
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pandas as pd
import random


INSTRUCTION = 'Determine the change or the status of the pulmonary edema.; '


device = 'cuda'
llm2vec_path = "microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned"
# af = pd.read_csv('/data/csv/candidate.csv')

# config = AutoConfig.from_pretrained("/data/research/model/llm2vec/8b/checkpoint-5779/")
text_model = LLM2Vec.from_pretrained(
    base_model_name_or_path='microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned',
    enable_bidirectional=True,
    pooling_mode="latent_attention",
    max_length=512,
    torch_dtype=torch.bfloat16,
)
ckpt = torch.load("/data/research/model/llm2vec/final/checkpoint-4896/pytorch_model.bin")
text_model.load_state_dict(ckpt, strict=False)

#text_model to bfloat
text_model.to(torch.bfloat16)
device = 'cpu'
text_model.to(device)
text_model.eval()
def tokenize(texts, tokenizer, max_length):
    texts_2 = []
    original_texts = []
    for text in texts:
        t = text.split('!@#$%^&*()')
        texts_2.append(t[1] if len(t) > 1 else "")
        original_texts.append("".join(t))

    original = tokenizer(
        original_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if 1:
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original
    

tokenizer = text_model.tokenizer
tokenizer.padding_side = "left"

df = pd.read_csv('/data/csv/llm2clip/nips/mimic_test2.csv')
ran = random.randint(0, len(df))
key = df['caption'][ran]
print(key)
key = INSTRUCTION + '!@#$%^&*()' + key
all_keys = [key] + ['No edema', 'Edema is improving', 'Edema is stable', 'Edema is worsening', 'Edema is present']
tokenized = tokenize(all_keys, tokenizer, 512).to(device)
tokenized = tokenized.to(torch.bfloat16)
emb = text_model(tokenized)
similarity = torch.nn.functional.cosine_similarity(emb[0], emb[1:], dim=1)
print(similarity)