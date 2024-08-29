import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from map import Map
from genetic import Genetic
from rescuer import Rescuer
from vs.constants import VS


class RescuerBoss(Rescuer):
    def __init__(self, env, config_file, rescuer_list):
        super().__init__(env, config_file)

        self.inactive_exp_counter = 0
        self.map = Map()
        self.map_victims = {}
        self.rescuers = rescuer_list
        self.rescuers.append(self)
        self.victim_clusters: list = [None, None, None, None]
        self.genetic = None
        self.cluster_order = {}

    def alert_explorer_inactive(self, map, victims):
        self.map_victims.update(victims)
        self.map.extend_map(map)
        self.inactive_exp_counter += 1

        if self.inactive_exp_counter == 4:
            self.prepare_rescuers()

    def prepare_rescuers(self) -> None:
        print(f"Mapa encontrado contendo {len(self.map.positions)} posições")
        self.genetic = Genetic(self.map, self.map_victims, self)
        self.cluster_victims()
        self.set_state(VS.ACTIVE)
        for i, r in enumerate(self.rescuers):
            r.go_save_victims(self.map, self.cluster_order[i])

    def cluster_victims(self) -> None:
        coordenadas = np.array([self.map_victims[id][0] for id in self.map_victims])
        features = np.array([self.map_victims[id][1] for id in self.map_victims])

        km = KMeans(n_clusters=4)
        km.fit_predict(coordenadas)

        dados_clusterizados = pd.DataFrame({'ID': list(self.map_victims.keys()),
                                            'Coordenadas': coordenadas.tolist(),
                                            'Features': features.tolist(),
                                            'Cluster': km.labels_})

        for cluster_id in range(4):
            cluster_data = dados_clusterizados[dados_clusterizados['Cluster'] == cluster_id]
            cluster_data.to_csv(f'cluster{cluster_id + 1}.csv', index=False)
            self.victim_clusters[cluster_id] = cluster_data.to_dict()
            self.cluster_order[cluster_id] = self.genetic.run(cluster_id + 1)[0]["victims"]
