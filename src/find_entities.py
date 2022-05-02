# %%
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


class NerResults():
    """
    For a given text, return these attributes:
        Find the person entities in the text
        Find the organisation entities in the text
        Find the location entities in the text
        Find the miscellaneous entities in the text
    """

    def __init__(self,
                text,
                 model,
                 tokenizer,
                 device):

        self.model = model
        self.tokenizer = tokenizer
        self.text = text
        self.device = device
        # first 20 words of the text
        self.text_head = self.text.split()[:20]
        self.ner_results = self.get_ner_results()
        self.ner_df = pd.DataFrame(self.ner_results)
        self.entities = ['PER', 'ORG', 'LOC', 'MISC']
        try:
            self.person_words = self.get_entity_words(self.entities[0])
            self.organisation_words = self.get_entity_words(self.entities[1])
            self.location_words = self.get_entity_words(self.entities[2])
            self.misc_words = self.get_entity_words(self.entities[3])
        except Exception as e:
            self.person_words = ['No person entities found']
            self.organisation_words = ['No organisation entities found']
            self.location_words = ['No location entities found']
            self.misc_words = ['No miscellaneous entities found']
            print(f"No entities found for {self.text_head}", e)


    def get_ner_results(self):
        """
        Get the NER results for a given text

        Parameters
        ----------
            text (str): the input texts

        Returns
        -------
            ner_results (list): the NER results a list of dictionaries

        """
        ner = pipeline('ner', model=self.model, tokenizer=self.tokenizer)
        return ner(self.text, show_tokens=False)

    def get_entity_df(self, entity):
        """
        Get the entity dataframe for a given entity

        Parameters
        ----------
            entity (str): the entity code

        Returns
        -------
            entity_df (pandas.DataFrame): the word and entity dataframe
                                         for that entity

        """
        df = self.ner_df[self.ner_df['entity']
                         .str.contains(f'-{entity}')].copy()
        return df[['word', 'entity']]

    def get_entity_index(self, entity):
        """
        Get the entity index for each entity begining code

        Parameters
        ----------
            entity (str): the entity code

        Returns
        -------
            entity_index (list): the entity index for that entity

        """
        df = self.get_entity_df(entity)
        return df[df['entity'] == f'B-{entity}'].index.tolist()

    def get_entity_words(self, entity):
        """
        Get the list of entity words for a given entity type code
        The identified entity words are returned broken down by
        begining (B-), middle (I-) prefixs.
        Loop through the results df and join them up

        Parameters
        ----------
            entity (str): the entity code

        Returns
        -------
            entity_words (list): the list of words for that entity

        """

        df = self.get_entity_df(entity)

        word_list = []
        begin_index = self.get_entity_index(entity)

        for i in range(len(begin_index)):
            if i < len(begin_index) - 1:
                word_parts_list = df.loc[begin_index[i]: begin_index[i+1]-1,
                                         'word'].tolist()
                # the middle of words is often marked by ##
                word = [''.join([x.strip('##') for x in word_parts_list])]
                word_list.append(word[0])
            else:
                # when we get an index larger than the length of the list
                break
        return word_list


if __name__ == "__main__":

    model = AutoModelForTokenClassification.from_pretrained('./models/dslim/bert-base-NER')
    tokenizer = AutoTokenizer.from_pretrained('./models/dslim/bert-base-NER')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data
    with open('../text_data/cycling_article.txt', 'r') as f:
        text = f.read()

    ner = NerResults(text, model, tokenizer, device)

    assert len(ner.ner_results) > 0
    assert isinstance(ner.ner_df, pd.core.frame.DataFrame)
    assert 'BernardHinault' in ner.person_words
    # TODO write more tests and move to the test folder

# %%
