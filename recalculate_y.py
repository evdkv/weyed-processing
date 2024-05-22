'''
File containing a script to recalculate
the y-coordinates of the labels to match the
x coordinate scale
'''

import json

with open('processed/info.json', 'r') as jsonl:
    participant_data = json.loads(jsonl.read())
    new_dict = {}
    with open("results/metadata.json", "r") as f:
        with open("processed/new_info.json", "w") as n:
            meta = json.load(f)
            for result in meta["data"][0]["studyResults"]:
                try:
                    id = result["urlQueryParameters"]["participant_id"]
                except KeyError:
                    id = "undefined"

                path = "results" + result["componentResults"][0]["path"] + "/data.txt"
                with open(path, "r") as f:
                    data = json.load(f)

                    height = data[0]["meta"]["window_innerHeight"]
                    width = data[0]["meta"]["window_innerWidth"]

                    ratio = height / width
                    print(ratio)

                    new_dict[id] = []
                    for example in participant_data[id]:
                        label = example["label"]
                        label[1] = label[1] * ratio
                        example["label"] = label
                        new_dict[id].append(example)
                        
            json.dump(new_dict, n)

                

