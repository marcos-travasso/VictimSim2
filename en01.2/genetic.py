import csv
import random
from datetime import datetime
import tensorflow as tf
import joblib
import pandas as pd

from read_map import read_matrix, shortest_path

victims = {}
BASE = (65, 22)
TIMELIMIT = 1000
CLUSTER = 4
POPULATION = 10_000
GENERATIONS = 1_001
MUTATION_RATE = 0.4
WORST_POP_CHANCE = 0.25
POPULATION_PERCENTAGE = 10
matrix = read_matrix(90, 90, '../datasets/data_300v_90x90/env_obst.txt')

MAXIMUNS = {'qPA': 8.733333, 'pulso': 199.889794, 'fResp': 21.996464}
MINIMUNS = {'qPA': -8.733333, 'pulso': 0.014052, 'fResp': 0.002916}


def normalize_values(qPA, pulso, fResp):
    normalized_qPA = (qPA - MINIMUNS["qPA"]) / (MAXIMUNS["qPA"] - MINIMUNS["qPA"])
    normalized_pulso = (pulso - MINIMUNS["pulso"]) / (MAXIMUNS["pulso"] - MINIMUNS["pulso"])
    normalized_fResp = (fResp - MINIMUNS["fResp"]) / (MAXIMUNS["fResp"] - MINIMUNS["fResp"])

    return normalized_qPA, normalized_pulso, normalized_fResp


gravity_cache = {}
def cached_gravity_cost(feat):
    if feat in gravity_cache:
        return gravity_cache[feat]
    grav = predict_grav_nn(feat[0], feat[1], feat[2])
    gravity_cache[feat] = grav
    return grav


path_cache = {}
def cached_shortest_path(start, end):
    if (start, end) in path_cache:
        return path_cache[(start, end)]
    distance, _ = shortest_path(matrix, start, end)
    path_cache[(start, end)] = distance
    return distance


def get_score_id(victims):
    return '-'.join([str(v['id']) for v in victims])


score_cache = {}
def get_list_score(victims):
    score_id = get_score_id(victims)
    if score_id in score_cache:
        return score_cache[score_id]
    path_cost = 0
    grav_cost = 0
    score = 0
    last_pos = BASE

    out_of_time_victims = []

    for victim in victims:
        c = victim["coords"]

        distance = cached_shortest_path(last_pos, c)
        last_pos = c
        # log(f"{last_pos} -> {c} = {distance}")
        path_cost += distance + 1

        grav_cost += victim["grav"] * path_cost

        if not can_return(c, path_cost):
            score += victim["grav"]
            out_of_time_victims.append(victim)

    distance = cached_shortest_path(last_pos, BASE)
    path_cost += distance
    score += path_cost + grav_cost
    # log(out_of_time_victims)
    score_cache[score_id] = score
    return score


def can_return(current_pos, current_time):
    distance = cached_shortest_path(current_pos, BASE)
    return (current_time + distance) <= TIMELIMIT


def run_population():
    population = []
    for i in range(POPULATION):
        shuffled_list = list(victims.keys())
        random.shuffle(shuffled_list)
        victims_list = [victims[x] for x in shuffled_list]
        score = get_list_score(victims_list)
        population.append({
            "id": i,
            "score": score,
            "victims": victims_list
        })
    population.sort(key=lambda x: x['score'])
    return population


def calculate_score(population):
    for i, p in enumerate(population):
        p['score'] = get_list_score(p['victims'])
    population.sort(key=lambda x: x['score'])


def predict_grav_nn(qPA, pulso, fResp):
    model = tf.keras.models.load_model('vital_signs_model.h5')

    scaler = joblib.load('scaler.save')

    input_data = pd.DataFrame([[qPA, pulso, fResp]], columns=['qPA', 'pulso', 'fResp'])

    input_data_scaled = scaler.transform(input_data)

    predicted_grav = model.predict(input_data_scaled)

    return predicted_grav[0][0]


def run_gen(gen, population=None):
    if not population:
        population = run_population()
    else:
        calculate_score(population)

    if gen % 100 == 0:
        csv_population = []

        for p in population:
            csv_population.append({
                "id": p['id'],
                "score": p['score'],
                "victims": ' '.join([str(victim['id']) for victim in p['victims']])
            })
        with open(f'./cluster{CLUSTER}/cluster{CLUSTER}-population-gen{gen}.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["id", "score", "victims"])
            writer.writeheader()
            writer.writerows(csv_population)

    best_pop = population[:int(len(population) / POPULATION_PERCENTAGE)]
    avg_pop = population[int(len(population) / POPULATION_PERCENTAGE):len(population) - int(len(population) / POPULATION_PERCENTAGE)]
    worst_pop = population[len(population) - int(len(population) / POPULATION_PERCENTAGE):]

    selected = []
    selected.extend(best_pop)
    selected.extend(avg_pop)

    for p in worst_pop:
        if random.random() > 1 - WORST_POP_CHANCE:
            selected.append(p)

    crossovered = crossover_population(selected, len(population) - len(best_pop))
    crossovered.extend(best_pop)

    mutated = mutate_population(crossovered)

    selected.extend(mutated)
    calculate_score(selected)
    return selected[:POPULATION]


def mutate_population(population):
    mutated_population = []

    for individual in population:
        if random.random() < MUTATION_RATE:
            victims = individual['victims']
            if len(victims) > 1:
                idx1, idx2 = random.sample(range(len(victims)), 2)
                victims[idx1], victims[idx2] = victims[idx2], victims[idx1]

            mutated_individual = {'id': individual['id'], 'victims': victims}
            mutated_population.append(mutated_individual)
        else:
            mutated_population.append(individual)

    return mutated_population


def order_crossover(parent1, parent2):
    size = len(parent1['victims'])

    start, end = sorted(random.sample(range(size), 2))

    child1_victims = [None] * size
    child2_victims = [None] * size

    child1_victims[start:end] = parent1['victims'][start:end]
    child2_victims[start:end] = parent2['victims'][start:end]

    fill_positions = [i for i in range(size) if child1_victims[i] is None]
    fill_values = [v for v in parent2['victims'] if v not in child1_victims]
    for pos, val in zip(fill_positions, fill_values):
        child1_victims[pos] = val

    fill_positions = [i for i in range(size) if child2_victims[i] is None]
    fill_values = [v for v in parent1['victims'] if v not in child2_victims]
    for pos, val in zip(fill_positions, fill_values):
        child2_victims[pos] = val

    return {'id': random.randint(0, 1e6), 'victims': child1_victims}, \
        {'id': random.randint(0, 1e6), 'victims': child2_victims}


def crossover_population(population, total):
    offspring = []

    while len(offspring) < total:
        parent1, parent2 = random.sample(population, 2)

        child1, child2 = order_crossover(parent1, parent2)

        offspring.append(child1)
        offspring.append(child2)

    return offspring[:total]


def print_ids(population):
    log(' '.join([str(p['id']) for p in population]))


def log(msg):
    print(f"[{datetime.now()}] {msg}")


with open(f'cluster{CLUSTER}.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        if row[0] == "ID":
            continue
        victim = {"id": int(row[0]), "coords": eval(f"{row[1:3]}".replace('"', "").replace("'", ""))[0]}
        victim["coords"] = (victim["coords"][0] + BASE[0], victim["coords"][1] + BASE[1])
        feat = eval(f"{row[3:9]}".replace('"', "").replace("'", ""))[0]
        victim["feat"] = normalize_values(feat[3], feat[4], feat[5])
        victim["grav"] = cached_gravity_cost(victim["feat"])
        victims[victim["id"]] = victim

pop = None
for i in range(GENERATIONS):
    if i % 10 == 0:
        log(f"Running generation {i}")
    pop = run_gen(i, pop)
