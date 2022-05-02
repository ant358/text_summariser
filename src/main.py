
# %%
import glob
import tqdm
import torch
import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime
from summarise_text import TextSummary
# turn off the deprecation warnings
import warnings
warnings.filterwarnings("ignore")

# get todays date and format for the file name
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# ask for the path to the text files to analyse
path = input("Enter the path to the text files: ") or "../text_data/"
# get the text files
text_files = glob.glob(path + "/*.txt")

# load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./models/t5-large')
tokenizer = T5Tokenizer.from_pretrained('./models/t5-large')
# set the device to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create an empty dataframe to store the results
results = pd.DataFrame(columns=["filename",
                                "char_count",
                                "word_count",
                                "summary"])

# loop through the files and summarize them
for file in tqdm.tqdm(text_files):
    with open(file, 'r') as f:

        text = TextSummary(f.read(),
                           model,
                           tokenizer,
                           device,
                           150)

        print("\n", file.split('\\')[-1].split(".")[0],
                # get the number of words in the text
                f"which has {text.len_words} words",
                "\nSummarized text: \n",
                text.text_summary)

        # add the results to the dataframe
        results = results.append({"filename": file.split('\\')[-1].split(".")[0],
                                  "char_count": text.len_char,
                                  "word_count": text.len_words,
                                  "summary": text.text_summary},
                                 ignore_index=True)

# check if the output directory exists
if not os.path.exists("../output"):
    os.makedirs("../output")

# save the results to a csv file
results.to_csv(f"../output/results_{now}.csv", index=False)
print(f"Job Done!\nResults saved to ../output/results_{now}.csv")
# %%
