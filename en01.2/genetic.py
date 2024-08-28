import csv
import random
from datetime import datetime

import joblib
import pandas as pd
import tensorflow as tf

from map import Map


class Genetic:
    TIMELIMIT = 1_000
    POPULATION = 1_000
    GENERATIONS = 1_001
    MUTATION_RATE = 0.4
    WORST_POP_CHANCE = 0.25
    POPULATION_PERCENTAGE = 10

    MAXIMUNS = {'qPA': 8.733333, 'pulso': 199.889794, 'fResp': 21.996464}
    MINIMUNS = {'qPA': -8.733333, 'pulso': 0.014052, 'fResp': 0.002916}

    def __init__(self, map, victims, rescuer):
        self.map: Map = map
        self.victims = {}
        self.rescuer = rescuer
        self.BASE = self.map.get_or_create((0, 0))

        for k, v in victims.items():
            self.victims[k] = {"id": k, "coords": self.map.get_or_create(v[0])}
            self.victims[k]["grav"] = self.cached_gravity_cost(self.normalize_values(v[1][3], v[1][4], v[1][5]))
            self.victims[k]["label"] = get_label(self.victims[k]["grav"])

    def run(self, cluster):
        pop = None
        for i in range(self.GENERATIONS):
            if i % 100 == 0:
                log(f"Cluster {cluster} | Running generation {i}")
            pop = self.run_gen(i, cluster, pop)
        return pop

    def normalize_values(self, qPA, pulso, fResp):
        normalized_qPA = (qPA - self.MINIMUNS["qPA"]) / (self.MAXIMUNS["qPA"] - self.MINIMUNS["qPA"])
        normalized_pulso = (pulso - self.MINIMUNS["pulso"]) / (self.MAXIMUNS["pulso"] - self.MINIMUNS["pulso"])
        normalized_fResp = (fResp - self.MINIMUNS["fResp"]) / (self.MAXIMUNS["fResp"] - self.MINIMUNS["fResp"])

        return normalized_qPA, normalized_pulso, normalized_fResp

    gravity_cache = {}

    def cached_gravity_cost(self, feat):
        if feat in self.gravity_cache:
            return self.gravity_cache[feat]
        grav = self.predict_grav_nn(feat[0], feat[1], feat[2])
        self.gravity_cache[feat] = grav
        return grav

    path_cache = {}

    def cached_shortest_path(self, start, end):
        if (start, end) in self.path_cache:
            return self.path_cache[(start, end)]

        distance = self.map.cost_path(start, end, self.rescuer)
        self.path_cache[(start, end)] = distance
        return distance

    score_cache = {}

    def get_list_score(self, victims):
        score_id = get_score_id(victims)
        if score_id in self.score_cache:
            return self.score_cache[score_id]
        path_cost = 0
        grav_cost = 0
        score = 0
        last_pos = self.BASE

        out_of_time_victims = []

        for victim in victims:
            c = victim["coords"]

            distance = self.cached_shortest_path(last_pos, c)
            last_pos = c
            path_cost += distance + 1

            grav_cost += victim["grav"] * path_cost

            if not self.can_return(c, path_cost):
                score += victim["grav"]
                out_of_time_victims.append(victim)

        distance = self.cached_shortest_path(last_pos, self.BASE)
        path_cost += distance
        score += path_cost + grav_cost
        self.score_cache[score_id] = score
        return score

    def can_return(self, current_pos, current_time):
        distance = self.cached_shortest_path(current_pos, self.BASE)
        return (current_time + distance) <= self.TIMELIMIT

    def run_population(self):
        population = []
        for i in range(self.POPULATION):
            shuffled_list = list(self.victims.keys())
            random.shuffle(shuffled_list)
            victims_list = [self.victims[x] for x in shuffled_list]
            score = self.get_list_score(victims_list)
            population.append({
                "id": i,
                "score": score,
                "victims": victims_list
            })
        population.sort(key=lambda x: x['score'])
        return population

    def calculate_score(self, population):
        for i, p in enumerate(population):
            p['score'] = self.get_list_score(p['victims'])
        population.sort(key=lambda x: x['score'])

    def predict_grav_nn(self, qPA, pulso, fResp):
        model = tf.keras.models.load_model('vital_signs_model.h5')

        scaler = joblib.load('scaler.save')

        input_data = pd.DataFrame([[qPA, pulso, fResp]], columns=['qPA', 'pulso', 'fResp'])

        input_data_scaled = scaler.transform(input_data)

        predicted_grav = model.predict(input_data_scaled)

        return predicted_grav[0][0]

    def run_gen(self, gen, cluster, population=None):
        if not population:
            population = self.run_population()

        if gen % 100 == 0:
            write_pop_to_file(cluster, gen, population)

        best_pop = population[:int(len(population) / self.POPULATION_PERCENTAGE)]
        avg_pop = population[int(len(population) / self.POPULATION_PERCENTAGE):len(population) - int(
            len(population) / self.POPULATION_PERCENTAGE)]
        worst_pop = population[len(population) - int(len(population) / self.POPULATION_PERCENTAGE):]

        selected = []
        selected.extend(best_pop)
        selected.extend(avg_pop)

        for p in worst_pop:
            if random.random() > 1 - self.WORST_POP_CHANCE:
                selected.append(p)

        crossovered = crossover_population(selected, len(population) - len(best_pop))
        crossovered.extend(best_pop)

        mutated = self.mutate_population(crossovered)

        selected.extend(mutated)
        self.calculate_score(selected)
        return selected[:self.POPULATION]

    def mutate_population(self, population):
        mutated_population = []

        for individual in population:
            if random.random() < self.MUTATION_RATE:
                victims = individual['victims']
                if len(victims) > 1:
                    idx1, idx2 = random.sample(range(len(victims)), 2)
                    victims[idx1], victims[idx2] = victims[idx2], victims[idx1]

                mutated_individual = {'id': individual['id'], 'victims': victims}
                mutated_population.append(mutated_individual)
            else:
                mutated_population.append(individual)

        return mutated_population


def print_ids(population):
    log(' '.join([str(p['id']) for p in population]))


def log(msg):
    print(f"[{datetime.now()}] {msg}")


def get_score_id(victims):
    return '-'.join([str(v['id']) for v in victims])


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


def write_pop_to_file(cluster, gen, population):
    csv_population = []
    for p in population:
        csv_population.append({
            "id": p['id'],
            "score": p['score'],
            "victims": ' '.join([str(victim['id']) for victim in p['victims']])
        })
    with open(f'./cluster{cluster}/cluster{cluster}-population-gen{gen}.csv', mode='w',
              newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "score", "victims"])
        writer.writeheader()
        writer.writerows(csv_population)


def get_label(grav):
    if grav <= 0.25:
        return 1
    elif grav <= 0.5:
        return 2
    elif grav <= 0.75:
        return 3
    return 4
