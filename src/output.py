# %%
import torch
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from transformers import T5Tokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


class SumResults():
    """
    Summarize text articles

    Parameters
    ----------
    text (str): the text to be summarized
    model (T5ForConditionalGeneration): the model to be used
    tokenizer (T5Tokenizer): the tokenizer to be used
    device (torch.device): the device to be used
    sum_length (int): the length of the summary

    Returns
    -------
    returns: the summary (str)
    returns: the length of the text in words (int)

    """

    def __init__(self, text, model, tokenizer, device, sum_length):

        self.model = model
        # self.display_architecture=False
        self.tokenizer = tokenizer
        self.text = text
        # remove the new line characters
        self.preprocess_text = text.strip().replace("\n", "")
        # how many words are in the text
        self.text_length = len(self.preprocess_text.split())
        self.sum_len = sum_length
        self.device = device
        self.summary = self.summarize()
        self.logger = logging.getLogger(__name__)

    def summarize(self) -> str:
        """
        The function takes in a text and the max
        length of the summary. It returns a summary.

        Returns
        -------
        returns: the summary
        """
        # add the prefix to the text
        t5_prepared_Text = f"summarize: {self.preprocess_text}"
        # encode the text
        tokenized_text = self.tokenizer.encode(
            t5_prepared_Text,
            return_tensors="pt",
            # truncate long sentences >512
            truncation=True).to(self.device)
        # submit the text to the model and adjust the parameters
        summary_ids = self.model.generate(
            tokenized_text,
            num_beams=4,
            no_repeat_ngram_size=2,
            min_length=30,
            max_length=self.sum_len,
            early_stopping=True)
        # decode the ids to text
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def load_to_graph_db(docment: dict[str, str], sum_results: SumResults):
    """
    Load the summary to the graph database
    Parameters
    ----------
        docment (dict): the document dictionary
        sum_results (SumResults): the summary results

    Returns
    -------
        None

    """
    # create the graph database connection
    with GraphDatabase.driver("bolt://host.docker.internal:7687") as driver:
        graph = driver.session()
        # summary and document length
        summary = sum_results.summary
        text_length = sum_results.text_length

        query = (
            "MATCH (d:Document {pageId: $document_id}) "
            "SET d.summary = $summary "
            "SET d.word_count = $text_length "
            )
        try:
            # create the entity node and the relationship
            result = graph.run(
                query,
                document_id=docment['pageId'],
                summary=summary,
                text_length=text_length
                )
            logger.info(
                f"Added summary information to document {docment['pageId']} - {result}"
            )
        except ServiceUnavailable as e:
            logger.exception(e)
            logger.error(
                f"During {docment['pageid']} could not connect to the graph database to add summary information"
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

    # load the model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('../models/t5-large')
    tokenizer = T5Tokenizer.from_pretrained('../models/t5-large')
    # try cpu first its probably enough for this example 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = SumResults(text, model, tokenizer, device, 50)

    print(summary.summarize())
    print(summary.text_length)
# %%
