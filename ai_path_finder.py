import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq
import time

# =====================================================
# PATHFINDER CLASS
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


# =====================================================
# DRAW GRID
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
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(color="black", linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    placeholder.pyplot(fig)
    plt.close(fig)


# =====================================================
# STREAMLIT APP
# =====================================================

st.set_page_config(page_title="Pathfinder", layout="wide")

if "grid_size" not in st.session_state:
    st.session_state.grid_size = 10
    st.session_state.walls = np.zeros((10, 10), dtype=int)

with st.sidebar:
    st.header("Configuration")

    size = st.slider("Grid Size", 5, 25, 10)

    if size != st.session_state.grid_size:
        st.session_state.grid_size = size
        st.session_state.walls = np.zeros((size, size), dtype=int)

    speed = st.slider("Animation Speed", 0.01, 0.2, 0.05)

    if st.button("Clear Grid"):
        st.session_state.walls = np.zeros((size, size), dtype=int)
        st.rerun()

    if st.button("Start Simulation", type="primary"):
        st.session_state.run = True


col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Start / Target")

    sx = st.number_input("Start Row", 0, size - 1, 0)
    sy = st.number_input("Start Col", 0, size - 1, 0)
    tx = st.number_input("Target Row", 0, size - 1, size - 1)
    ty = st.number_input("Target Col", 0, size - 1, size - 1)

    start = (sx, sy)
    target = (tx, ty)

    st.session_state.walls[start] = 0
    st.session_state.walls[target] = 0

    st.divider()
    st.subheader("Wall Editor")

    wall_input = st.text_input("Enter position (row,col)", placeholder="Example: 4,5")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Add Wall"):
            try:
                r, c = map(int, wall_input.split(","))
                if (r, c) == start or (r, c) == target:
                    st.warning("Cannot place wall on start/target.")
                else:
                    st.session_state.walls[r, c] = 1
                    st.rerun()
            except:
                st.error("Invalid format. Use row,col")

    with col_b:
        if st.button("Remove Wall"):
            try:
                r, c = map(int, wall_input.split(","))
                st.session_state.walls[r, c] = 0
                st.rerun()
            except:
                st.error("Invalid format. Use row,col")

    st.divider()

    walls_list = [
        f"({r},{c})"
        for r in range(size)
        for c in range(size)
        if st.session_state.walls[r, c] == 1
    ]

    if walls_list:
        st.write("Current Walls:")
        st.text(", ".join(walls_list))
    else:
        st.info("No walls placed yet.")


with col2:
    placeholder = st.empty()
    draw_grid(st.session_state.walls, start, target, [], [], start, placeholder)

    if st.session_state.get("run", False):

        pf = Pathfinder(st.session_state.walls, start, target)
        path, visited = pf.bfs()

        if not path:
            st.error("No path found.")
        else:
            for pos in path:
                draw_grid(st.session_state.walls, start, target, visited, path, pos, placeholder)
                time.sleep(speed)

            st.success(f"Reached Goal in {len(path)-1} steps.")

        st.session_state.run = False
