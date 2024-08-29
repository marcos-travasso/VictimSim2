import csv
import random
from datetime import datetime
import tensorflow as tf
import joblib
import pandas as pd

from read_map import read_matrix, shortest_path

matrix = read_matrix(94, 94, '../datasets/data_408v_94x94/env_obst.txt')
TIMELIMIT = 100
victims = {}
BASE = (46,46)
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

clusters = set()
for c in range(1, 5):
    best_pop = None
    with open(f'cluster{c}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if row[0] == "ID":
                continue
            victim = {"id": int(row[0]), "coords": eval(f"{row[1:3]}".replace('"', "").replace("'", ""))[0]}
            victim["coords"] = (victim["coords"][0] + BASE[0], victim["coords"][1] + BASE[1])
            feat = eval(f"{row[3:9]}".replace('"', "").replace("'", ""))[0]
            victim["feat"] = normalize_values(feat[3], feat[4], feat[5])
            feat = victim["feat"]
            victim["grav"] = predict_grav_nn(feat[0], feat[1], feat[2])
            victims[victim["id"]] = victim
    with open(f'cluster{c}/cluster{c}-population-gen1000.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if row[0] == "id":
                continue
            best_pop = row
            break

    seq = best_pop[3].split(" ")
    seq = [victims[int(v)] for v in seq]


    def can_return(current_pos, current_time):
        distance, _ = shortest_path(matrix, current_pos, BASE)
        return (current_time + distance) <= TIMELIMIT

    last_pos = BASE
    cost = 0
    saved = []
    for v in seq:
        distance, _ = shortest_path(matrix, last_pos, v["coords"])
        if not can_return(v["coords"], cost):
            break
        last_pos = v["coords"]
        clusters.add(v["grav"])
        cost += distance

    print(f"cluster {c} ({len(saved)}): {saved}")
    # clusters.append(saved)
    print("_----------------")

print(clusters)

a = [[{'id': 314, 'coords': (74, 73), 'feat': (0.6012263588254335, 0.3812587572532939, 0.7417812260213767),
       'grav': 0.767441}],
     [{'id': 92, 'coords': (39, 45), 'feat': (0.5, 0.42473227191321694, 0.7166214382508906), 'grav': 0.88616604},
      {'id': 176, 'coords': (36, 47), 'feat': (0.17577515938073127, 0.2548559644621607, 0.14009049381209435),
       'grav': 0.34490702},

      {'id': 177, 'coords': (36, 51), 'feat': (0.5, 0.4491106679669011, 0.6487690389927082), 'grav': 0.8625903},
      {'id': 178, 'coords': (36, 56), 'feat': (0.5096126530386509, 0.5027442249595251, 0.7599740614838497),
       'grav': 0.7686463},
      {'id': 189, 'coords': (39, 62), 'feat': (0.5, 0.31535035402145, 0.29440975144165005), 'grav': 0.62981427},
      {'id': 201, 'coords': (46, 72), 'feat': (0.35789388770587355, 0.30115452929750725, 0.3636686995658909),
       'grav': 0.51945496},
      {'id': 197, 'coords': (44, 76), 'feat': (0.755292223484436, 0.2821288488324911, 0.7329110791946801),
       'grav': 0.5921271},
      {'id': 195, 'coords': (42, 76), 'feat': (0.004367576502579294, 0.2550953531919846, 0.2805978371475125),
       'grav': 0.29020914},
      {'id': 171, 'coords': (33, 81), 'feat': (1.0, 0.8387133842384936, 0.9119097109752369), 'grav': 0.17458394}],
     [{'id': 91, 'coords': (39, 38), 'feat': (0.5, 0.4352408708006197, 0.8714450710726618), 'grav': 0.82275736},
      {'id': 90, 'coords': (36, 39), 'feat': (0.46364011311603487, 0.3748851173745736, 0.7080860259563395),
       'grav': 0.85143894},
      {'id': 88, 'coords': (34, 37), 'feat': (0.541984715342928, 0.3342015010505877, 0.8375170299944329),
       'grav': 0.7454963},
      {'id': 87, 'coords': (33, 34), 'feat': (0.5, 0.39084010004575737, 0.8946305070923527), 'grav': 0.8290515},
      {'id': 86, 'coords': (26, 34), 'feat': (0.5, 0.42155161080027403, 0.692976231029209), 'grav': 0.89097667},
      {'id': 83, 'coords': (22, 34), 'feat': (0.6422674481781468, 0.4472148351049023, 0.7458462363598635),
       'grav': 0.7222012},
      {'id': 71, 'coords': (23, 31), 'feat': (0.002977213854092125, 0.3158267349921833, 0.6674042769270333),
       'grav': 0.5987091},
      {'id': 70, 'coords': (20, 32), 'feat': (0.39066991949121826, 0.17104269211418363, 0.6368802341486695),
       'grav': 0.5899648},
      {'id': 64, 'coords': (7, 28), 'feat': (0.7671756018005955, 0.3187616484245497, 0.6643685229868322),
       'grav': 0.6474602}],
     [{'id': 226, 'coords': (56, 38), 'feat': (0.5, 0.3788835515617498, 0.7300620618374078), 'grav': 0.864838},
      {'id': 255, 'coords': (69, 30), 'feat': (0.6690722774455068, 0.4250280256620635, 0.7046211461652299),
       'grav': 0.75573426},
      {'id': 254, 'coords': (69, 28), 'feat': (0.6622169909243126, 0.3824862448790809, 0.717927866845313),
       'grav': 0.76314294},
      {'id': 267, 'coords': (73, 23), 'feat': (0.32503283683331435, 0.3939065852223327, 0.9325738166484098),
       'grav': 0.75565094},
      {'id': 271, 'coords': (74, 24), 'feat': (0.5, 0.4062071424355237, 0.6919837581457979), 'grav': 0.8892655},
      {'id': 288, 'coords': (82, 26), 'feat': (0.5, 0.4048367009939605, 0.6840391554832354), 'grav': 0.8876873}]]
