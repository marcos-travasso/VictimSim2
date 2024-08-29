import csv
import ast
import csv
import random
from datetime import datetime
import tensorflow as tf
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

for i in range(1, 5):
    order_mapping = {}
    with open(f'./cluster{i}/cluster{i}-population-gen1000.csv', mode='r') as order_file:
        order_reader = csv.DictReader(order_file)
        for row in order_reader:
            victims = list(map(int, row['victims'].split()))
            for index, victim_id in enumerate(victims):
                order_mapping[victim_id] = index

    data = []
    with open(f'./cluster{i}.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            victim_id = int(row['ID'])
            coordinates = ast.literal_eval(row['Coordenadas'])
            x = int(coordinates[0])
            y = int(coordinates[1])
            if victim_id in order_mapping:
                feat = eval(row["Features"])
                feat = normalize_values(feat[3], feat[4], feat[5])
                grav = predict_grav_nn(feat[0], feat[1], feat[2])
                label = 1 if grav < 0.25 else 2 if grav < 0.5 else 3 if grav < 0.75 else 4
                data.append((order_mapping[victim_id], victim_id, x, y, grav, label))

    data.sort()

    with open(f'cluster{i}-p.csv', mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        for _, victim_id, x, y, g, z in data:
            writer.writerow([victim_id, x, y, g, z])