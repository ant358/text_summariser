import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from ..src.summarise_text import TextSummary

# load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('../src/models/t5-large')
tokenizer = T5Tokenizer.from_pretrained('../src/models/t5-large')
# set the device to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text = """  In all criminal prosecutions, the accused shall enjoy the right to a
speedy and public trial, by an impartial jury of the State and district
wherein the crime shall have been committed, which district shall have
been previously ascertained by law, and to be informed of the nature
and cause of the accusation; to be confronted with the witnesses against him;
to have compulsory process for obtaining witnesses in his favor,
and to have the assistance of counsel for his defense.
"""
tx = TextSummary(text, model, tokenizer, device, max_length=50)
print("Text Summary:\n", tx.text_summary)

def test_len_char():
    assert tx.len_char == 574

def test_len_words():
    assert tx.len_words == 81

def test_text_summary_min_length():
    assert len(tx.text_summary) >= 30

def test_text_summary_type():
    assert isinstance(tx.text_summary, str)