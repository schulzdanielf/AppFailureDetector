import json


def update_parameters(dict):
    with open("params/best_hyper.json", "w") as fp:
        json.dump(dict, fp)


def get_best(key):
    f = open("params/best_hyper.json")
    data = json.load(f)
    f.close()
    return data[key]
