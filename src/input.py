# %%
import requests
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

logger = logging.getLogger(__name__)


# import ids from the text source database
def get_document(pageid: str) -> dict[str, str]:
    """
    Get a document from the text database
    send a GET request to the text database
    using a pageid to get the document text
    and title.

    Returns:
        dict[str, str]: json with the keys
                        pageid, title and text
    """
    try:
        response = requests.get(
            f"http://host.docker.internal:8080/return_article/{pageid}")
        if response.status_code == 200:
            return requests.get(
                f"http://host.docker.internal:8080/return_article/{pageid}"
            ).json()
        logger.error(
            f"Could not get {pageid} from text database - response code {response.status_code}"
        )
        return {"pageid": pageid, "title": "Error", "text": "Error"}
    except requests.exceptions.ConnectionError:
        logger.exception(f"Could not return {pageid} text database")
        return {}


# provide some text to the NER model
def text_input() -> dict[str, str]:
    return {
        "pageid":
            "1",
        "title":
            "Cylcing News - Liege-Bastogne-Liege",
        "text":
            """The oversized snowflakes fell softly and silently, settling among the pines like a picturesque Christmas scene.
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
    }


# query the graph database to get nodes without a entity relationship
def get_pageids_from_graph() -> list[str]:
    """
    Get the pageids from the graph database
    without a summary  -query the graph database
    to get the pageids

    Returns:
        list[str]: list of pageids without a summary
    """
    try:
        driver = GraphDatabase.driver("bolt://host.docker.internal:7687")
        with driver.session() as session:
            result = session.run("MATCH(d:Document) WHERE d.summary is NULL RETURN d.pageId")
            return [record['n.pageId'] for record in result]
    except ServiceUnavailable:
        logger.error("Could not connect to the graph database")
        return []
