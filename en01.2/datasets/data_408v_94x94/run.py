import csv

import tensorflow as tf
import numpy as np
import joblib
import pandas as pd



def predict_grav_nn(qPA, pulso, fResp):
    model = tf.keras.models.load_model('vital_signs_model.h5')

    scaler = joblib.load('scaler.save')

    input_data = pd.DataFrame([[qPA, pulso, fResp]], columns=['qPA', 'pulso', 'fResp'])

    input_data_scaled = scaler.transform(input_data)

    predicted_grav = model.predict(input_data_scaled)

    return predicted_grav[0][0]

MAXIMUNS = {'qPA': 8.733333, 'pulso': 199.889794, 'fResp': 21.996464}
MINIMUNS = {'qPA': -8.733333, 'pulso': 0.014052, 'fResp': 0.002916}

def normalize_values(qPA, pulso, fResp):
    normalized_qPA = (qPA - MINIMUNS["qPA"]) / (MAXIMUNS["qPA"] - MINIMUNS["qPA"])
    normalized_pulso = (pulso - MINIMUNS["pulso"]) / (MAXIMUNS["pulso"] - MINIMUNS["pulso"])
    normalized_fResp = (fResp - MINIMUNS["fResp"]) / (MAXIMUNS["fResp"] - MINIMUNS["fResp"])

    return normalized_qPA, normalized_pulso, normalized_fResp

def predict_grav_nn(qPA, pulso, fResp):
    model = tf.keras.models.load_model('vital_signs_model.h5')

    scaler = joblib.load('scaler.save')

    input_data = pd.DataFrame([[qPA, pulso, fResp]], columns=['qPA', 'pulso', 'fResp'])

    input_data_scaled = scaler.transform(input_data)

    predicted_grav = model.predict(input_data_scaled)

    return predicted_grav[0][0]


# predicted_grav = predict_grav_nn(qPA, pulso, fResp)
# print(f'Predicted grav NN: {predicted_grav}')
data = []
with open("env_vital_signals_cego.txt", mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        # Convert each element in the row to float
        feat = [float(item) for item in row]
        idd = int(feat[0])
        feat = normalize_values(feat[3], feat[4], feat[5])
        grav = predict_grav_nn(feat[0], feat[1], feat[2])
        label = 1 if grav < 0.25 else 2 if grav < 0.5 else 3 if grav < 0.75 else 4
        data.append((idd, grav * 100, label))

print(data)
with open("result_cego.csv", mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    for row in data:
        csv_writer.writerow(row)