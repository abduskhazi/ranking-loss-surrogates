import pickle
import sys
from study_hpo import convert_to_json

#name = "./Q1_Research/ListWise_transfer/1/results/DE_LIST_32x32x10_E1000_l0_02_OPT"
name = sys.argv[1]
dest_name = sys.argv[2]

def store_object(obj, obj_name):
    with open(obj_name, "wb") as fp:
        pickle.dump(obj, fp)

def load_object(obj_name):
    with open(obj_name, "rb") as fp:
        return pickle.load(fp)

performance = []
for i in range(430):
    partial = load_object(name + str(i))
    performance += partial

store_object(performance, dest_name)
convert_to_json(dest_name)
