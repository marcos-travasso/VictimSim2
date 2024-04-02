from math import sqrt
from random import randint

from vs.constants import VS
import heapq

class Map:
    def __init__(self):
        self.positions = {}

    def in_map(self, coord):
        return coord in self.positions

    def size(self):
        return len(self.positions)

    def get_or_create(self, coord):
        if self.in_map(coord):
            pos = self.get(coord)
        else:
            pos = Position(coords=coord, visited=False)
            self.add(pos)
        return pos
    def get(self, coord):
        return self.positions.get(coord)

    def add(self, position):
        if self.visited(position.coords) and self.get(position.coords).difficulty > position.difficulty:
            return

        self.positions[position.coords] = position
        return position

    def visited(self, coord):
        return coord in self.positions and self.positions[coord].visited

    def get_closest_not_visited(self, pos):
        queue = list(pos.neighborhood.values())
        verified = {}
        while len(queue) != 0:
            p = queue.pop()
            if not p.visited and p.difficulty <= 3:
                return p

            if p.coords not in verified:
                queue.extend(list(p.neighborhood.values()))
                verified[p.coords] = True
        return None

    def time_to_return(self, actual_pos, explorer):
        path = self.get_path(actual_pos, self.get((0,0)), explorer)[1:]
        time = 0
        before = actual_pos
        for p in path:
            dx, dy = before.coords[0] - p.coords[0], before.coords[1] - p.coords[1]
            cost = explorer.COST_LINE if dx == 0 or dy == 0 else explorer.COST_DIAG
            time += p.difficulty * cost
        return time

    def get_path(self, actual_pos, wanted_pos, explorer):
        # print("")
        # print(f"{actual_pos} atual = {actual_pos.visited} indo para: {wanted_pos} com {wanted_pos.visited}")
        open_list = []
        best_for = {}
        closed_set = set()

        def heuristic(node):
            return abs(node.coords[0] - wanted_pos.coords[0]) + abs(node.coords[1] - wanted_pos.coords[1]) * explorer.COST_LINE

        g_score = {actual_pos: 0}
        f_score = {actual_pos: heuristic(actual_pos)}
        parent = {actual_pos: None}
        heapq.heappush(open_list, (f_score[actual_pos], actual_pos))

        while open_list:
            # print("open list")
            _, current_node = heapq.heappop(open_list)

            if current_node == wanted_pos:
                # print("ACHEI AQUI DENTRRO")
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = parent[current_node]
                return path[::-1]

            closed_set.add(current_node)

            for neighbor in current_node.neighborhood.values():
                if neighbor in closed_set:
                    continue

                dx, dy = current_node.coords[0] - neighbor.coords[0], current_node.coords[1] - neighbor.coords[1]
                cost = explorer.COST_LINE if dx == 0 or dy == 0 else explorer.COST_DIAG

                tentative_g_score = g_score[current_node] + (neighbor.difficulty * cost)
                if neighbor not in open_list or tentative_g_score < g_score[neighbor]:
                    parent[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                    if neighbor not in open_list and (neighbor not in best_for or best_for[neighbor] > f_score[neighbor]):
                        best_for[neighbor] = f_score[neighbor]
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None

    def draw(self):
        print("TODO o draw")
        return
        if not self.map_data:
            print("Map is empty.")
            return

        min_x = min(key[0] for key in self.map_data.keys())
        max_x = max(key[0] for key in self.map_data.keys())
        min_y = min(key[1] for key in self.map_data.keys())
        max_y = max(key[1] for key in self.map_data.keys())

        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                item = self.get((x, y))
                if item:
                    if item[1] == VS.NO_VICTIM:
                        row += f"[{item[0]:7.2f}  no] "
                    else:
                        row += f"[{item[0]:7.2f} {item[1]:3d}] "
                else:
                    row += f"[     ?     ] "
            print(row)

class Position:
    def __init__(self, coords, visited, difficulty = 3):
        self.coords = coords
        self.difficulty = difficulty
        self.victim_seq = VS.NO_VICTIM
        self.visited = visited
        self.neighborhood = {}

    def __str__(self):
        return f"({self.coords[0]},{self.coords[1]})"

    def __lt__(self, other):
        return self.difficulty < other.difficulty
    # def __str__(self):
    #     return f"({self.coords[0]}, {self.coords[1]}) - Difficulty: {self.difficulty} - Visited: {self.visited} - Neighborhood: {len(self.neighborhood)} - Victim Seq {self.victim_seq}"