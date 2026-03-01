import tkinter as tk
import random
import time
import heapq
import math

# ---------------- CONFIG ----------------
CELL_SIZE = 25
OBSTACLE_PROBABILITY = 0.03  # dynamic obstacle spawn chance per step

# ---------------- NODE CLASS ----------------
class Node:
    def __init__(self, r, c):
        self.r = r
        self.c = c
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

# ---------------- HEURISTICS ----------------
def manhattan(a, b):
    return abs(a.r - b.r) + abs(a.c - b.c)

def euclidean(a, b):
    return math.sqrt((a.r - b.r)**2 + (a.c - b.c)**2)

# ---------------- MAIN APP ----------------
class PathfindingApp:
    def __init__(self, root):
        self.root = root
        root.title("Dynamic Pathfinding Agent")

        self.rows = 15
        self.cols = 20

        self.algorithm = tk.StringVar(value="A*")
        self.heuristic = tk.StringVar(value="Manhattan")
        self.dynamic_mode = tk.BooleanVar(value=False)

        self.create_ui()
        self.create_grid()

    # ---------- UI ----------
    def create_ui(self):
        control = tk.Frame(self.root)
        control.pack(side=tk.TOP, fill=tk.X)

        tk.Label(control, text="Algorithm").pack(side=tk.LEFT)
        tk.OptionMenu(control, self.algorithm, "A*", "GBFS").pack(side=tk.LEFT)

        tk.Label(control, text="Heuristic").pack(side=tk.LEFT)
        tk.OptionMenu(control, self.heuristic, "Manhattan", "Euclidean").pack(side=tk.LEFT)

        tk.Checkbutton(control, text="Dynamic Mode", variable=self.dynamic_mode).pack(side=tk.LEFT)

        tk.Button(control, text="Random Map", command=self.random_map).pack(side=tk.LEFT)
        tk.Button(control, text="Start Search", command=self.start_search).pack(side=tk.LEFT)

        self.metrics = tk.Label(self.root, text="Nodes: 0 | Cost: 0 | Time: 0 ms")
        self.metrics.pack()

    # ---------- GRID ----------
    def create_grid(self):
        self.canvas = tk.Canvas(
            self.root,
            width=self.cols * CELL_SIZE,
            height=self.rows * CELL_SIZE,
            bg="white"
        )
        self.canvas.pack()

        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)

        self.draw_grid()
        self.canvas.bind("<Button-1>", self.toggle_wall)

    def draw_grid(self):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                color = "white"
                if self.grid[r][c] == 1:
                    color = "black"
                if (r, c) == self.start:
                    color = "green"
                if (r, c) == self.goal:
                    color = "red"

                self.canvas.create_rectangle(
                    c * CELL_SIZE, r * CELL_SIZE,
                    (c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE,
                    fill=color, outline="gray"
                )

    def toggle_wall(self, event):
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if (r, c) not in [self.start, self.goal]:
            self.grid[r][c] ^= 1
            self.draw_grid()

    def random_map(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in [self.start, self.goal]:
                    self.grid[r][c] = 1 if random.random() < 0.3 else 0
        self.draw_grid()

    # ---------- SEARCH ----------
    def start_search(self):
        start_time = time.time()
        path, visited = self.search()
        exec_time = int((time.time() - start_time) * 1000)

        if path:
            for r, c in path:
                self.canvas.create_rectangle(
                    c * CELL_SIZE, r * CELL_SIZE,
                    (c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE,
                    fill="green"
                )

        self.metrics.config(
            text=f"Nodes: {visited} | Cost: {len(path)} | Time: {exec_time} ms"
        )

    def search(self):
        open_set = []
        visited_set = set()

        start_node = Node(*self.start)
        goal_node = Node(*self.goal)

        start_node.g = 0
        start_node.h = self.get_heuristic(start_node, goal_node)
        start_node.f = start_node.h

        heapq.heappush(open_set, (start_node.f, start_node))
        visited_count = 0

        nodes = {(start_node.r, start_node.c): start_node}

        while open_set:
            _, current = heapq.heappop(open_set)
            visited_count += 1

            if (current.r, current.c) == self.goal:
                return self.reconstruct(current), visited_count

            visited_set.add((current.r, current.c))
            self.color_cell(current.r, current.c, "blue")

            if self.dynamic_mode.get():
                self.spawn_dynamic_obstacles()

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = current.r + dr, current.c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 1:
                    continue

                neighbor = nodes.get((nr, nc), Node(nr, nc))
                nodes[(nr, nc)] = neighbor

                tentative_g = current.g + 1

                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.get_heuristic(neighbor, goal_node)

                    if self.algorithm.get() == "A*":
                        neighbor.f = neighbor.g + neighbor.h
                    else:
                        neighbor.f = neighbor.h

                    if (neighbor.r, neighbor.c) not in visited_set:
                        heapq.heappush(open_set, (neighbor.f, neighbor))
                        self.color_cell(neighbor.r, neighbor.c, "yellow")

            self.root.update()
            time.sleep(0.03)

        return [], visited_count

    def reconstruct(self, node):
        path = []
        while node:
            path.append((node.r, node.c))
            node = node.parent
        return path[::-1]

    def get_heuristic(self, a, b):
        if self.heuristic.get() == "Manhattan":
            return manhattan(a, b)
        return euclidean(a, b)

    def spawn_dynamic_obstacles(self):
        if random.random() < OBSTACLE_PROBABILITY:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if (r, c) not in [self.start, self.goal]:
                self.grid[r][c] = 1
                self.color_cell(r, c, "black")

    def color_cell(self, r, c, color):
        self.canvas.create_rectangle(
            c * CELL_SIZE, r * CELL_SIZE,
            (c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE,
            fill=color
        )

# ---------------- RUN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingApp(root)
    root.mainloop()
    
 


