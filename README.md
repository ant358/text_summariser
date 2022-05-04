# Text Analyser

Takes in a text file and outputs its attributes  

## Installation
Clone the repository

Download and install python 3.9.7 make sure it is added to the path

Create a virtual environment with the right version of python and install the dependencies:

`py -3.9 -m venv txt_sum_venv`

`txt_sum_venv` is already in `.gitignore` - if you change the name then you need to update the  `.gitignore` file

activate the virtual environment
`source txt_sum_venv/bin/activate` or on windows  
`source txt_sum_venv/Scripts/activate`  

Next install the dependencies:  
`pip install -r requirements.txt`

Next install the models  
run the get_models.py script to download the models they need to be saved in `src/models/`

Put data to analyse in `./text_data/`

Run `./src/main.py`

Input which folder the text files are in. This is a placeholder to later code the docker volume.  

For running in a Docker container:  
`docker build -t txt_sum_img .`  
`docker run -it  txt_sum_img bash`  
then  
 `python3 main.py`  

Still working out the how to work with volumes and docker.

To run the tests  
`python -m pytest` in the root dir  
