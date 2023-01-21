# %%
import torch
import pandas as pd
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

logger = logging.getLogger(__name__)


class NerResults():
    """
    For a given text, return these attributes:
        Find the person entities in the text
        Find the organisation entities in the text
        Find the location entities in the text
        Find the miscellaneous entities in the text

    Parameters
    ----------
        text (str): the input text
        model (transformers.modeling_auto.AutoModelForTokenClassification):
            the NER model
        tokenizer (transformers.tokenization_auto.AutoTokenizer): the NER
            tokenizer
        device (torch.device): the device to use

    Attributes
    ----------
        ner_results (list): the raw NER results a list of dictionaries
        ner_df (pd.DataFrame): the NER results as a dataframe
        entities (list): the entities types for reference
        unquie_entities (pd.DataFrame): the unique entities found in the text
    """

    def __init__(self, text, model, tokenizer, device):

        self.model = model
        self.tokenizer = tokenizer
        self.text = text
        self.device = device
        self.aggregation_strategy = 'first'
        self.logger = logging.getLogger(__name__)
        self.ner_results = self.get_ner_results()
        # print(self.ner_results)
        self.ner_df = pd.DataFrame(self.ner_results)
        print(self.ner_df.head())
        self.entities = ['PER', 'ORG', 'LOC', 'MISC']
        self.unique_entities = self.get_unique_entities(self.ner_df)
        # print(self.unique_entities)

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
        try:
            ner = pipeline(
                'ner',
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy=self.aggregation_strategy)
            self.logger.info(f"Getting NER results for {self.text}")
            return ner(self.text)
        except Exception as e:
            self.logger.exception(e)
            return []

    def get_entity_df(self, ner_results):
        """
        Get the entity dataframe for the results
        Parameters
        ----------
            ner_results (list): the NER results a list of dictionaries

        Returns
        -------
            entity_df (pandas.DataFrame): the word and entity dataframe
                                         for that entity

        """
        try:
            return pd.DataFrame(ner_results)
        except Exception as e:
            self.logger.exception(e)
            return pd.DataFrame()

    def get_unique_entities(self, df):
        """
        Get the unique entities in the dataframe
        Parameters
        ----------
            df (pandas.DataFrame): the entity dataframe

        Returns
        -------
            unique_entities (pandas.DataFrame): the unique entities in the dataframe

        """
        try:
            return df.groupby(['entity_group', 'word'], as_index=False).agg({
                'score': 'mean',
                'start': 'unique',
                'end': 'unique'
            })
        except Exception as e:
            self.logger.exception(e)
            return pd.DataFrame()


def load_to_graph_db(docment: dict[str, str], ner_results: pd.DataFrame):
    """
    Load the document and the NER results to the graph database
    Parameters
    ----------
        docment (dict): the document dictionary
        ner_results (list): the NER results a list of dictionaries

    Returns
    -------
        None

    """
    # create the graph database connection
    with GraphDatabase.driver("bolt://host.docker.internal:7687") as driver:
        graph = driver.session()
        # for each row of the dataframe, create a node for the entity
        # print(ner_results.head())
        # and a relationship between the entity and the documentd
        for index, row in ner_results.iterrows():
            # get the entity attributes
            entity_group = row['entity_group']
            word = row['word']
            score = row['score']
            start = str(row['start'])
            end = str(row['end'])

            query = (
                "MATCH (d:Document {pageId: $document_id}) "
                "MERGE (e:Entity {name: $word}) "
                "MERGE (d)-[:HAS_ENTITY {score: $score, start: $start, end: $end}]->(e) "
                "MERGE (t:EntityType {name: $entity_group}) "
                "MERGE (e)-[:IS_A]->(t)"
                )
            try:
                # create the entity node and the relationship
                result = graph.run(
                    query,
                    document_id=docment['pageId'],
                    entity_group=entity_group,
                    word=word,
                    score=score,
                    start=start,
                    end=end)
                logger.info(
                    f"Created entity node for {word} in document {docment['pageId']} - {result}"
                )
            except ServiceUnavailable as e:
                logger.exception(e)
                logger.error(
                    f"During {docment['pageid']} could not connect to the graph database for entity creation"
                )


if __name__ == "__main__":

    text = """The oversized snowflakes fell softly and silently, settling among the pines like a picturesque Christmas scene.
    By the roadside, spectators in heavy winter coats watched team cars and motorbikes struggle up one of Liege-Bastogne-Liege's countless climbs, tyres spinning in the slush as they pursued one man on a bike.
    It was April 1980 and Bernard Hinault, almost unrecognisable beneath a big red balaclava, slewed doggedly on, further into the lead, somehow remaining balanced on the two wheels beneath him.
    He was under such physical strain that he would do himself permanent damage. Pushing his body to its very limit, he raced through the Ardennes in search of victory in the race known as 'La Doyenne' - the old lady.
    So bad were the conditions that several of cycling's best riders collected their number from organisers and then never lined up.
    After just 70km of the 244km one-day race, 110 of the 174 entrants were already holed up in a hotel by the finish line. Only 21 completed the course. Hinault suffered frostbite.
    Rarely do you see such attrition in cycling, but Liege-Bastogne-Liege, which celebrates its 130th birthday on Sunday, has been making and breaking the toughest competitors for years.
    Hinault was 25. He had already won the Tour de France twice and would go on to win it a further three times, an icon of his sport in the making. His total of five Tour victories remains a joint record.
    But this was a different challenge - a long way from the searing heat and sunflowers of summer.
    One of the five prestigious 'Monument' one-day races in cycling, Liege-Bastogne-Liege is celebrated by many for being the very antithesis of the Tour.
    In the hills of east and south Belgium the peloton is stretched through thick, damp forest, over short, sharp climbs and across tricky, part-cobbled sections before landing back where it all began in Liege.
    "[The race is] already hard, it's long, and when I won it was in very tough conditions, especially the snow," says Hinault, now aged 67.
    "Yes, I considered quitting if the weather conditions persisted. We started having difficulties. It's difficult in Liege-Bastogne-Liege."
    Hinault's account of one of his greatest triumphs is characteristically taciturn. Tough conditions is a severe understatement. And in the racing he didn't have it all his own way, either.
    With around 91km to go, approaching the 500m Stockeu climb, Rudy Pevenage was two minutes 15 seconds ahead of Hinault and a small chasing group.
    Pevenage was one of the hard men of the spring classics. He was a Belgian with a big lead, in conditions many locals would feel only a Belgian could master.
    But even he did not finish a race that truly separated the men from the legends. 'Neige-Bastogne-Neige,' as it would be dubbed.
    On the next climb, a 500m ascent of the Haute Levee, Hinault and a small number of fellow pursuers caught up with Pevenage. Then Hinault launched his attack, bright red balaclava and thick blue gloves disappearing into the distance as his stunning acceleration left everybody behind.
    There were still 80km to go.
    """

    model = AutoModelForTokenClassification.from_pretrained(
        '../models/dslim/bert-base-NER')
    tokenizer = AutoTokenizer.from_pretrained('../models/dslim/bert-base-NER')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ner = NerResults(text, model, tokenizer, device)

    assert len(ner.ner_results) > 0
    assert isinstance(ner.ner_df, pd.core.frame.DataFrame)
    # assert 'Bernard Hinault' in ner.person_words
    # TODO write more tests and move to the test folder

# %%
