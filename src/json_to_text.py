# %%
import json
import glob

# find json files
json_files = glob.glob("../text_data/json/*/*.json")

for file in json_files:

    # load the json file
    with open(file, 'r', encoding='UTF-8') as f:
        data = json.load(f)
        for i in range(10):  # len(data['items'])):
            data_id = data['items'][i]['id']
            data_name = data['items'][i]['name']
            data_text = data['items'][i]['properties']['text']
            print(data_id, data_name, data_text)

# %%
