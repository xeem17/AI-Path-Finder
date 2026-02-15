import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq
import time

# =====================================================
# 1. PATHFINDER CLASS (No Yield – Fast Execution)
# =====================================================

class Pathfinder:
    def __init__(self, grid, start, target):
        self.grid = grid
        self.start = start
        self.target = target
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

    def is_safe(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def get_neighbors(self, node):
        r, c = node
        neighbors = []
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if self.is_safe(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def build_path(self, came_from, current):
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        return path

    # ---------------- BFS ----------------
    def bfs(self):
        queue = collections.deque([self.start])
        came_from = {self.start: None}
        visited = {self.start}

        while queue:
            current = queue.popleft()

            if current == self.target:
                return self.build_path(came_from, current), visited

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

        return None, visited

    # ---------------- DFS ----------------
    def dfs(self):
        stack = [self.start]
        came_from = {self.start: None}
        visited = {self.start}

        while stack:
            current = stack.pop()

            if current == self.target:
                return self.build_path(came_from, current), visited

            for neighbor in reversed(self.get_neighbors(current)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append(neighbor)

        return None, visited

    # ---------------- UCS ----------------
    def ucs(self):
        pq = [(0, self.start)]
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}
        visited = set()

        while pq:
            cost, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == self.target:
                return self.build_path(came_from, current), visited

            for neighbor in self.get_neighbors(current):
                move_cost = 1.414 if (
                    neighbor[0] != current[0] and neighbor[1] != current[1]
                ) else 1.0

                new_cost = cost_so_far[current] + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))

        return None, visited

    # ---------------- DLS ----------------
    def dls(self, limit):
        stack = [(self.start, 0)]
        came_from = {self.start: None}
        visited = {self.start}

        while stack:
            current, depth = stack.pop()

            if current == self.target:
                return self.build_path(came_from, current), visited

            if depth < limit:
                for neighbor in reversed(self.get_neighbors(current)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        came_from[neighbor] = current
                        stack.append((neighbor, depth + 1))

        return None, visited

    # ---------------- IDDFS ----------------
    def iddfs(self):
        for depth in range(1, 50):
            path, visited = self.dls(depth)
            if path:
                return path, visited
        return None, visited

    # ---------------- Bidirectional ----------------
    def bidirectional(self):
        q_start = collections.deque([self.start])
        q_end = collections.deque([self.target])

        parent_start = {self.start: None}
        parent_end = {self.target: None}

        visited_start = {self.start}
        visited_end = {self.target}

        while q_start and q_end:

            current = q_start.popleft()
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited_start:
                    visited_start.add(neighbor)
                    parent_start[neighbor] = current
                    q_start.append(neighbor)

                    if neighbor in visited_end:
                        return self.merge_paths(parent_start, parent_end, neighbor), visited_start | visited_end

            current = q_end.popleft()
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited_end:
                    visited_end.add(neighbor)
                    parent_end[neighbor] = current
                    q_end.append(neighbor)

                    if neighbor in visited_start:
                        return self.merge_paths(parent_start, parent_end, neighbor), visited_start | visited_end

        return None, visited_start | visited_end

    def merge_paths(self, p_start, p_end, meet):
        path1 = self.build_path(p_start, meet)
        path2 = []
        node = p_end[meet]
        while node:
            path2.append(node)
            node = p_end[node]
        return path1 + path2


# =====================================================
# 2. DRAW GRID (Optimized)
# =====================================================

def draw_grid(walls, start, target, visited, path, agent, placeholder):
    fig, ax = plt.subplots(figsize=(6, 6))

    rows, cols = walls.shape
    color_map = np.ones((rows, cols, 3))

    color_map[walls == 1] = [0.5, 0.0, 0.5]

    for r, c in visited:
        color_map[r, c] = [0.85, 0.92, 1.0]

    if path:
        for r, c in path:
            color_map[r, c] = [1.0, 0.8, 0.0]

    color_map[start] = [0.0, 0.8, 0.0]
    color_map[target] = [0.8, 0.0, 0.0]
    color_map[agent] = [0.0, 0.0, 1.0]

    ax.imshow(color_map)
    ax.set_xticks([])
    ax.set_yticks([])

    placeholder.pyplot(fig)
    plt.close(fig)


# =====================================================
# 3. STREAMLIT APP
# =====================================================

st.set_page_config(page_title="Pathfinder Optimized", layout="wide")

if "grid_size" not in st.session_state:
    st.session_state.grid_size = 10
    st.session_state.walls = np.zeros((10, 10), dtype=int)

with st.sidebar:
    st.header("Configuration")

    size = st.slider("Grid Size", 5, 25, 10)
    if size != st.session_state.grid_size:
        st.session_state.grid_size = size
        st.session_state.walls = np.zeros((size, size), dtype=int)

    algo = st.selectbox("Algorithm", ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"])
    dls_limit = st.slider("DLS Limit", 1, 30, 10)
    speed = st.slider("Animation Speed", 0.01, 0.2, 0.05)

    if st.button("Clear Grid"):
        st.session_state.walls = np.zeros((size, size), dtype=int)
        st.rerun()

    if st.button("Start Simulation", type="primary"):
        st.session_state.run = True

col1, col2 = st.columns([1, 1.5])

with col1:
    sx = st.number_input("Start X", 0, size - 1, 0)
    sy = st.number_input("Start Y", 0, size - 1, 0)
    tx = st.number_input("Target X", 0, size - 1, size - 1)
    ty = st.number_input("Target Y", 0, size - 1, size - 1)

start = (sx, sy)
target = (tx, ty)

st.session_state.walls[start] = 0
st.session_state.walls[target] = 0

with col2:
    placeholder = st.empty()

    draw_grid(st.session_state.walls, start, target, [], [], start, placeholder)

    if st.session_state.get("run", False):

        pf = Pathfinder(st.session_state.walls, start, target)

        if algo == "BFS":
            path, visited = pf.bfs()
        elif algo == "DFS":
            path, visited = pf.dfs()
        elif algo == "UCS":
            path, visited = pf.ucs()
        elif algo == "DLS":
            path, visited = pf.dls(dls_limit)
        elif algo == "IDDFS":
            path, visited = pf.iddfs()
        else:
            path, visited = pf.bidirectional()

        if not path:
            st.error("No path found.")
        else:
            for pos in path:
                draw_grid(st.session_state.walls, start, target, visited, path, pos, placeholder)
                time.sleep(speed)

            st.success(f"Reached Goal in {len(path)-1} steps!")
            st.session_state.run = False
