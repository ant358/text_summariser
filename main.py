# note does not run in jupyter notebook, run in the terminal
from fastapi import FastAPI
import uvicorn
import logging
import os
import pathlib
import torch
from datetime import datetime
from transformers import (T5Tokenizer, T5ForConditionalGeneration)
from src.output import SumResults, load_to_graph_db
from src.input import (get_document, get_pageids_from_graph, text_input)
from src.control import Job_list
from src.get_models import (get_t5_model, get_t5_tokenizer, save_model)

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
create_text_sum = Job_list()
# check the model is in the models folder
if not os.path.exists("./models/t5-large/config.json"):
    logger.info("t5 model not found, loading from huggingface")
    try:
        model = get_t5_model()
        tokenizer = get_t5_tokenizer()
        save_model(model, tokenizer, "./models/t5-large")
        logger.info("t5 model loaded from huggingface and saved")
    except Exception as e:
        logger.error(f"Error loading model from huggingface: {e}")
        raise SystemExit from e

# load the model
model = T5ForConditionalGeneration.from_pretrained('./models/t5-large')
tokenizer = T5Tokenizer.from_pretrained('./models/t5-large')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# status
status = "paused"    # paused, running, stopped


def update_jobs():
    """Get the pageids of nodes in the graph database
    that do not have a summary"""
    # get the pageids of nodes in the graph database
    # that do not have a sum result
    pageids = get_pageids_from_graph()
    # add the pageids to the job list
    create_text_sum.bulk_add(pageids)
    logger.info(f'{len(pageids)} Jobs added to the job list')


def run(model, tokenizer, device, sum_length=50):
    while len(create_text_sum) > 0:
        # get the first job
        job = create_text_sum.get_first_job()
        # get the document
        document = get_document(job)
        # run the model
        sum_result = SumResults(document['text'], model, tokenizer, device, sum_length)
        # save the results
        load_to_graph_db(document, sum_result)
        # log the results
        logger.info(f'Job {job} complete')


# OUTPUT- routes
@app.get("/")
async def root():
    logging.info("Root requested")
    return {"message": "text summary conatiner API to work with text data"}


@app.get("/get_current_jobs")
async def get_current_jobs():
    """Get the current jobs"""
    logging.info("Current jobs list requested")
    return {"Current jobs": create_text_sum.jobs}


@app.get("/example_summary_result")
async def example_sum_result():
    """Get an example of the summary result from the text database"""
    logging.info("Example summary result requested")
    result = get_document("18942")
    sum_result = SumResults(result['text'], model, tokenizer, device, 50)
    return {
        "Orginal Text": result['text'],
        "Word Count": sum_result.text_length,
        "Example Summary result": sum_result.summary
    }


@app.get("/test_summary_result/")
async def test_sum_result():
    """Get an example of the summary result from some sample text"""
    logging.info("Summary result test requested")
    result = text_input()
    sum_result = SumResults(result['text'], model, tokenizer, device, 50)
    return {
        "Orginal Text": result['text'],
        "Word Count": sum_result.text_length,
        "Example Summary result": sum_result.summary
    }


@app.get("/get_status")
async def get_status():
    """Get the status of the controller"""
    logging.info("Status requested")
    return {"Status": status}


# INPUT routes
@app.post("/add_job/{job}")
async def add_job(job: str):
    """Add a job to the list of jobs"""
    create_text_sum.add(job)
    run(model, tokenizer, device, 50)
    logging.info(f"Job {job} added")
    return {"message": f"Job {job} added"}


@app.post("/add_jobs_list/{jobs}")
async def add_jobs_list(jobs: str):
    """Add a list of jobs to the list of jobs"""
    create_text_sum.add_list(jobs)
    run(model, tokenizer, device, 50)
    logging.info(f"Jobs {jobs} added")
    return {"message": f"Jobs {jobs} added"}


@app.post("/remove_job/{job}")
async def remove_job(job: str):
    """Remove a job from the list of jobs"""
    create_text_sum.remove(job)
    logging.info(f"Job {job} removed")
    return {"message": f"Job {job} removed"}


@app.post("/remove_jobs_list/{jobs}")
async def remove_jobs_list(jobs: str):
    """Remove a list of jobs from the list of jobs"""
    create_text_sum.remove_list(jobs)
    logging.info(f"Jobs {jobs} removed")
    return {"message": f"Jobs {jobs} removed"}


@app.post("/remove_all_jobs")
async def remove_all_jobs():
    """Remove all jobs from the list of jobs"""
    create_text_sum.clear()
    logging.info("All jobs removed")
    return {"message": "All jobs removed"}


@app.post("/update_graph_summaries")
async def update_summary_jobs():
    """Check the graph for summaries and update the jobs list
    then run the jobs"""
    update_jobs()
    run(model, tokenizer, device, 50)
    logging.info("Jobs list updated")
    return {"message": "Jobs list updated summaries being created"}


if __name__ == "__main__":
    # goto localhost:9080/
    # or localhost:9080/docs for the interactive docs
    uvicorn.run(app, port=9080, host="0.0.0.0")
