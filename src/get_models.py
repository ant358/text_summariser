import torch
from transformers import (
                          T5Tokenizer,
                          T5ForConditionalGeneration,
                          AutoTokenizer,
                          AutoModelForTokenClassification
                          )

def get_t5_model():
# load the model and tokenizer
    return T5ForConditionalGeneration.from_pretrained('t5-large')

def get_t5_tokenizer():
    return T5Tokenizer.from_pretrained('t5-large')

def get_ner_tokenizer():
    return AutoTokenizer.from_pretrained('dslim/bert-base-NER')

def get_ner_model():
    return AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')

def save_model(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_t5_model()
    tokenizer = get_t5_tokenizer()
    save_model(model, tokenizer, '..src/models/t5-large')
    print('t5_model saved')

    model = get_ner_model()
    tokenizer = get_ner_tokenizer()
    save_model(model, tokenizer, '..src/models/dslim/bert-base-NER')
    print('ner_model saved')

