from .utils import get_data_path
import pandas as pd
import os
import pickle


def read_file(path=os.path.join(get_data_path(), "input", "<My_data>.parquet")):
    try:
        df = pd.read_parqet(path,engine = 'pyarrow')
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)

#
# def read_model(name="model.pkl"):
#     path = os.path.join(get_data_path(), "output", name)
#     with open(path, "rb") as f:
#         model = pickle.load(f)
#     return model

def read_predict_distance_nyc(name="predict_distance_nyc.pkl"):
    path = os.path.join(get_data_path(), "output", name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def read_predict_fare_nyc(name="predict_fare_nyc.pkl"):
    path = os.path.join(get_data_path(), "output", name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def read_predict_payment_type_nyc(name="predict_payment_type_nyc.pkl"):
    path = os.path.join(get_data_path(), "output", name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
