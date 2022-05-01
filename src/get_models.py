import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_t5_model():
# load the model and tokenizer
    return T5ForConditionalGeneration.from_pretrained('t5-large')


def get_t5_tokenizer():
    return T5Tokenizer.from_pretrained('t5-large')


def save_model(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_t5_model()
    tokenizer = get_t5_tokenizer()
    save_model(model, tokenizer, '..src/models/t5-large')
    print('model saved')

