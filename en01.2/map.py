import heapq
from collections import deque

from vs.constants import VS


class Position:
    def __init__(self, coords, visited, difficulty=3):
        self.coords = coords
        self.difficulty = difficulty
        self.victim_seq = VS.NO_VICTIM
        self.visited = visited
        self.neighborhood = {}
        self.action_seq = []

    def __str__(self):
        return f"({self.coords[0]},{self.coords[1]})"

    def __lt__(self, other):
        return self.difficulty < other.difficulty


class Map:
    def __init__(self):
        self.positions = {}

    def in_map(self, coord):
        return coord in self.positions

    def get_or_create(self, coord):
        if self.in_map(coord):
            pos = self.get(coord)
        else:
            pos = Position(coords=coord, visited=False)
            self.add(pos)
        return pos

    def get(self, coord) -> Position:
        return self.positions.get(coord)

    def add(self, position):
        if self.visited(position.coords) and self.get(position.coords).difficulty > position.difficulty:
            return

        self.positions[position.coords] = position
        return position

    def extend_map(self, new_map):
        repeated = set(self.positions.keys()).intersection(set(new_map.positions.keys()))

        for r in repeated:
            for n in new_map.positions[r].neighborhood.keys():
                if n not in self.positions[r].neighborhood:
                    nn = self.get_or_create(n)
                    self.positions[r].neighborhood[n] = nn
                    nn.neighborhood[self.positions[r].coords] = self.positions[r]

        for p in new_map.positions:
            if p not in self.positions:
                np = self.get_or_create(p)
                self.positions[p] = np
                for n in new_map.positions[p].neighborhood.keys():
                    if n not in np.neighborhood:
                        nn = self.get_or_create(n)
                        np.neighborhood[n] = nn
                        nn.neighborhood[np.coords] = np

    def visited(self, coord):
        return coord in self.positions and self.positions[coord].visited

    def get_closest_not_visited(self, pos):
        queue = list(pos.neighborhood.values())
        verified = {}
        while len(queue) != 0:
            p = queue.pop()
            if not p.visited:
                return p

            if p.coords not in verified:
                queue.extend(list(p.neighborhood.values()))
                verified[p.coords] = True
        return None

    def time_to_return(self, actual_pos, explorer):
        path = self.get_path(actual_pos, self.get((0, 0)), explorer)[1:]
        time = 0
        before = actual_pos
        for p in path:
            dx, dy = before.coords[0] - p.coords[0], before.coords[1] - p.coords[1]
            cost = explorer.COST_LINE if dx == 0 or dy == 0 else explorer.COST_DIAG
            time += p.difficulty * cost
        return time

    def cost_path(self, actual_pos, wanted_pos, explorer):
        path = self.get_path(actual_pos, wanted_pos, explorer)
        if path is None:
            raise Exception("Path not found")
        cost = 0
        before = actual_pos
        for p in path:
            dx, dy = before.coords[0] - p.coords[0], before.coords[1] - p.coords[1]
            cost += explorer.COST_LINE if dx == 0 or dy == 0 else explorer.COST_DIAG
            before = p
        return cost

    def get_path(self, actual_pos, wanted_pos, explorer):
        queue = deque([actual_pos])
        parent = {actual_pos: None}

        while queue:
            current_node = queue.popleft()

            if current_node == wanted_pos:
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = parent[current_node]
                return path[::-1]

            for neighbor in current_node.neighborhood.values():
                if neighbor not in parent:
                    parent[neighbor] = current_node
                    queue.append(neighbor)

        return None

    def draw(self):
        min_x = min(self.positions[key].coords[0] for key in self.positions.keys())
        max_x = max(self.positions[key].coords[0] for key in self.positions.keys())
        min_y = min(self.positions[key].coords[1] for key in self.positions.keys())
        max_y = max(self.positions[key].coords[1] for key in self.positions.keys())

        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                pos = self.get((x, y))
                if pos:
                    if pos.victim_seq == VS.NO_VICTIM:
                        row += f"[{pos.difficulty:7.2f}  no] "
                    else:
                        row += f"[{pos.difficulty:7.2f} {pos.victim_seq:3d}] "
                else:
                    row += f"[     ?     ] "
            print(row)
