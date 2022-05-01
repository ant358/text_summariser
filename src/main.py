
# %%
import glob
import tqdm
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from summarise_text import TextSummary
# turn off the deprecation warnings
import warnings
warnings.filterwarnings("ignore")


# ask for the path to the text files to analyse
path = input("Enter the path to the text files: ") or "../text_data/"
# get the text files
text_files = glob.glob(path + "/*.txt")

# load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./models/t5-large')
tokenizer = T5Tokenizer.from_pretrained('./models/t5-large')
# set the device to cpu
device = torch.device('cpu')

# loop through the files and summarize them
for file in tqdm.tqdm(text_files):
    with open(file, 'r') as f:

        text=TextSummary(f.read(), 
                          model, 
                          tokenizer, 
                          device,
                          150)

        print("\n", file.split('\\')[-1].split(".")[0], 
                # get the number of words in the text
                f"which has {text.len_words} words",
                "\nSummarized text: \n", 
                text.text_summary)
# %%
