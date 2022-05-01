
# %%
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
# turn off the deprecation warnings
import warnings
warnings.filterwarnings("ignore")



class TextSummary():
    """
    For a given text, return these attributes:
        Summarise the text
        Count the number of words in the text
        Count the number of characters in the text
    """

    def __init__(self, text, model, tokenizer, device, max_length):
        self.text = text
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.len_words = len(self.text.split())
        self.len_char = len(self.text.strip())
        self.ml = max_length
        self.prepared_text = self.preprocess_text_for_summariser()
        self.tokenized_text = self.encode_the_text()
        self.text_summary = self.summarize()


    def preprocess_text_for_summariser(self):
        """Strip out additional spaces and line returns
        Then prefix the text with summarize

        Parameters
        ----------
            text (str): the input texts

        Returns
        -------
            text (str) : preprocessed text
        """
        preprocess_text = self.text.strip().replace("\n","")
        # add the prefix to the text
        return "summarize: " + preprocess_text


    def encode_the_text(self):
        """
        Encode the text to ids

        Parameters
        ----------
            text (str): the input texts

        Returns
        -------
            text (torch.tensor): the encoded text

        """
        return self.tokenizer.encode(self.prepared_text, 
                                     return_tensors="pt",
                                     # truncate long sentences > 512
                                     truncation=True).to(self.device)



    def decode_the_summariser_tokens(self, summary_ids):
        """
        Decode the tokens from the model
        
        Parameters
        ----------
            summary_ids (torch.tensor): the summary tokens
            
        Returns
        -------
        summary (str) : the summary
            
        """
        # decode the ids to text
        output = self.tokenizer.decode(summary_ids[0], 
                                       skip_special_tokens=True)
        return output


    def summarize(self):
        """
        Generate the summary

        Parameters
        ----------
            text (str): the input texts

        Returns
        -------
            text (str) : the summary
        """
        # submit the text to the model and adjust the parameters 
        summary_ids =  self.model.generate(  
                                            self.tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=self.ml,
                                            early_stopping=True
                                            )
        
        return self.decode_the_summariser_tokens(summary_ids)


if __name__ == "__main__":

    # test the TextAnalysis class is working nb move to test folder later

    # load the model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('./models/t5-large')
    tokenizer = T5Tokenizer.from_pretrained('./models/t5-large')
    # set the device to cpu
    device = torch.device('cpu')

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

    assert tx.len_char == 574
    assert tx.len_words == 81
    assert len(tx.text_summary) >= 30
    assert isinstance(tx.text_summary, str)