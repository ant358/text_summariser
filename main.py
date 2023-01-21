# note does not run in jupyter notebook, run in the terminal
from fastapi import FastAPI
import uvicorn
import logging
import os
import pathlib
import torch
from datetime import datetime
from transformers import (AutoTokenizer, AutoModelForTokenClassification)
from src.output import NerResults, load_to_graph_db
from src.input import (get_document, get_pageids_from_graph,
                       get_entity_relationship_from_graph, text_input)
from src.control import Job_list
from src.get_models import get_ner_tokenizer, get_ner_model, save_model

# setup logging
# get todays date
datestamp = datetime.now().strftime('%Y%m%d')
container_name = os.getenv('CONTAINER_NAME')
# append date to logfile name
log_name = f'log-{container_name}-{datestamp}.txt'
path = os.path.abspath('./logs/')
# add path to log_name to create a pathlib object
# required for loggin on windows and linux
log_filename = pathlib.Path(path, log_name)

# create log file if it does not exist
if os.path.exists(log_filename) is not True:
    # create the logs folder if it does not exist
    if os.path.exists(path) is not True:
        os.mkdir(path)
    # create the log file
    open(log_filename, 'w').close()

# create logger
logger = logging.getLogger()
# set minimum output level
logger.setLevel(logging.DEBUG)
# Set up the file handler
file_logger = logging.FileHandler(log_filename)

# create console handler and set level to debug
ch = logging.StreamHandler()
# set minimum output level
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('[%(levelname)s] -'
                              ' %(asctime)s - '
                              '%(name)s : %(message)s')
# add formatter
file_logger.setFormatter(formatter)
ch.setFormatter(formatter)
# add a handler to logger
logger.addHandler(file_logger)
logger.addHandler(ch)
# mark the run
logger.info(f'Lets get started! - logginng in "{log_filename}" today')

# create the FastAPI app
app = FastAPI()

# create the job list
create_ner_nodes = Job_list()
# check the model is in the models folder
if not os.path.exists("./models/dslim/bert-base-NER/config.json"):
    model = get_ner_model()
    tokenizer = get_ner_tokenizer()
    save_model(model, tokenizer, "./models/dslim/bert-base-NER")
    logger.info("ner_model loaded from huggingface and saved")

# load the model
model = AutoModelForTokenClassification.from_pretrained(
    './models/dslim/bert-base-NER')
tokenizer = AutoTokenizer.from_pretrained('./models/dslim/bert-base-NER')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# status
status = "paused"    # paused, running, stopped


def update_jobs():
    """Get the pageids of nodes in the graph database
    that do not have a NER result"""
    # get the pageids of nodes in the graph database
    graph_pageids = get_pageids_from_graph()
    # that do not have a NER result
    nodes_with_a_ner = get_entity_relationship_from_graph()
    # add the pageids to the job list
    if pageids := [
            pageid for pageid in graph_pageids
            if pageid not in nodes_with_a_ner
    ]:
        create_ner_nodes.bulk_add(pageids)
        logger.info(f'{len(pageids)} Jobs added to the job list')


def run(model, tokenizer, device):
    while len(create_ner_nodes) > 0:
        # get the first job
        job = create_ner_nodes.get_first_job()
        # get the document
        document = get_document(job)
        # run the model
        entities = NerResults(document['text'], model, tokenizer, device)
        # save the results
        load_to_graph_db(document, entities.unique_entities)
        # log the results
        logger.info(f'Job {job} complete')


# OUTPUT- routes
@app.get("/")
async def root():
    logging.info("Root requested")
    return {"message": "text NER conatiner API to work with text data"}


@app.get("/get_current_jobs")
async def get_current_jobs():
    """Get the current jobs"""
    logging.info("Current jobs list requested")
    return {"Current jobs": create_ner_nodes.jobs}


@app.get("/example_ner_result")
async def example_ner_result():
    """Get an example of the NER result from the text database"""
    logging.info("Example NER result requested")
    result = get_document("18942")
    entities = NerResults(result['text'], model, tokenizer, device)
    df = entities.unique_entities
    return {"Example NER result": df.to_json()}


@app.get("/test_ner_result/")
async def test_ner_result():
    """Get an example of the NER result from some sample text"""
    logging.info("NER result test requested")
    result = text_input()
    entities = NerResults(result['text'], model, tokenizer, device)
    return {"Example NER result": entities.unique_entities.to_json()}


@app.get("/get_status")
async def get_status():
    """Get the status of the controller"""
    logging.info("Status requested")
    return {"Status": status}


# INPUT routes
@app.post("/add_job/{job}")
async def add_job(job: str):
    """Add a job to the list of jobs"""
    create_ner_nodes.add(job)
    run(model, tokenizer, device)
    logging.info(f"Job {job} added")
    return {"message": f"Job {job} added"}


@app.post("/add_jobs_list/{jobs}")
async def add_jobs_list(jobs: str):
    """Add a list of jobs to the list of jobs"""
    jobs.add_list(jobs)
    run(model, tokenizer, device)
    logging.info(f"Jobs {jobs} added")
    return {"message": f"Jobs {jobs} added"}


@app.post("/remove_job/{job}")
async def remove_job(job: str):
    """Remove a job from the list of jobs"""
    create_ner_nodes.remove(job)
    logging.info(f"Job {job} removed")
    return {"message": f"Job {job} removed"}


@app.post("/remove_jobs_list/{jobs}")
async def remove_jobs_list(jobs: str):
    """Remove a list of jobs from the list of jobs"""
    jobs.remove_list(jobs)
    logging.info(f"Jobs {jobs} removed")
    return {"message": f"Jobs {jobs} removed"}


@app.post("/remove_all_jobs")
async def remove_all_jobs():
    """Remove all jobs from the list of jobs"""
    create_ner_nodes.clear()
    logging.info("All jobs removed")
    return {"message": "All jobs removed"}


@app.post("/update_graph")
async def update_entity_jobs():
    """Check the graph for entity relationships and update the jobs list"""
    update_jobs()
    run(model, tokenizer, device)
    logging.info("Jobs list updated")
    return {"message": "Jobs list updated NER nodes being created"}


if __name__ == "__main__":
    # goto localhost:8080/
    # or localhost:8080/docs for the interactive docs
    uvicorn.run(app, port=7080, host="0.0.0.0")
