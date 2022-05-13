
# %%
import glob
import tqdm
import torch
import os
import pandas as pd
from transformers import (
                          T5Tokenizer,
                          T5ForConditionalGeneration,
                          AutoTokenizer,
                          AutoModelForTokenClassification
                          )
from datetime import datetime
from summarise_text import TextSummary
from find_entities import NerResults

# turn off the deprecation warnings
import warnings
warnings.filterwarnings("ignore")

# get todays date and format for the file name
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# ask for the path to the text files to analyse
path = input("Enter the path to the text files: ") or "../text_data/"
# get the text files
text_files = glob.glob(path + "/*.txt")

# load the summariser model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('./models/t5-large')
t5_tokenizer = T5Tokenizer.from_pretrained('./models/t5-large')

# load the NER model and tokenizer
ner_model = AutoModelForTokenClassification.from_pretrained(
                                              './models/dslim/bert-base-NER'
    )
ner_tokenizer = AutoTokenizer.from_pretrained('./models/dslim/bert-base-NER')

# set the device to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create an empty dataframe to store the results
results = pd.DataFrame(columns=["filename",
                                "char_count",
                                "word_count",
                                "summary",
                                "people",
                                "locations",
                                "organisations",
                                "misc entities"])

# loop through the files and summarize them
for file in tqdm.tqdm(text_files):
    with open(file, 'r') as f:
        try:
            # get the text
            txt = f.read()

            sum = TextSummary(text=txt,
                              model=t5_model,
                              tokenizer=t5_tokenizer,
                              device=device,
                              max_length=150)

            ents = NerResults(text=txt,
                              model=ner_model,
                              tokenizer=ner_tokenizer,
                              device=device)

            print("\n", file,  # file.split('\\')[-1].split(".")[0]
                  # get the number of words in the text
                  f"which has {sum.len_words} words",
                  "\nSummarized text: \n",
                  sum.text_summary,
                  f"\nPeople: {ents.person_words}",
                  f"\nLocations: {ents.location_words}",
                  f"\nOrganisations: {ents.organisation_words}",
                  f"\nOther Entities: {ents.misc_words}"
                  )

            # add the results to the dataframe
            results = results.append({
                                      "filename": file,
                                      # .split('\\')[-1].split(".")[0],
                                      "char_count": sum.len_char,
                                      "word_count": sum.len_words,
                                      "summary": sum.text_summary,
                                      "people": ents.person_words,
                                      "locations": ents.location_words,
                                      "organisations": ents.organisation_words,
                                      "misc entities": ents.misc_words,
                                     },
                                     ignore_index=True)
        except Exception as e:
            print(f"Error summarising text for {file}", e)

# check if the output directory exists
if not os.path.exists("../output"):
    os.makedirs("../output")

# save the results to a csv file
results.to_csv(f"../output/results_{now}.csv", index=False)
print(f"Job Done!\nResults saved to ../output/results_{now}.csv")
# %%
