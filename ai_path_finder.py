import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq
import time
import random

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

        # Strict clockwise order as per requirement:
        # 1. Up, 2. Right, 3. Bottom, 4. Bottom-Right, 5. Left, 6. Top-Left
        # (Excludes Top-Right and Bottom-Left diagonals)
        self.directions = [
            (-1, 0),   # Up
            (0, 1),    # Right
            (1, 0),    # Bottom
            (1, 1),    # Bottom-Right (Diagonal)
            (0, -1),   # Left
            (-1, -1)   # Top-Left (Diagonal)
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
                        return self.merge(parent_start, parent_end, neighbor), visited_start | visited_end

            current = q_end.popleft()
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited_end:
                    visited_end.add(neighbor)
                    parent_end[neighbor] = current
                    q_end.append(neighbor)

                    if neighbor in visited_start:
                        return self.merge(parent_start, parent_end, neighbor), visited_start | visited_end

        return None, visited_start | visited_end

    def merge(self, p1, p2, meet):
        path1 = self.build_path(p1, meet)
        path2 = []
        node = p2[meet]
        while node:
            path2.append(node)
            node = p2[node]
        return path1 + path2


# =====================================================
# DRAW GRID (WITH BORDERS)
# =====================================================

def draw_grid(walls, start, target, visited, path, agent, placeholder, dynamic_obstacles=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    rows, cols = walls.shape
    color_map = np.ones((rows, cols, 3))

    # Static walls (purple)
    color_map[walls == 1] = [0.5, 0.0, 0.5]

    # Dynamic obstacles (dark red)
    if dynamic_obstacles:
        for r, c in dynamic_obstacles:
            color_map[r, c] = [0.6, 0.0, 0.0]

    # Visited nodes (light blue)
    for r, c in visited:
        color_map[r, c] = [0.85, 0.92, 1.0]

    # Path (yellow/gold)
    if path:
        for r, c in path:
            color_map[r, c] = [1.0, 0.8, 0.0]

    # Start (green)
    color_map[start] = [0.0, 0.8, 0.0]
    # Target (red)
    color_map[target] = [0.8, 0.0, 0.0]
    # Agent (blue)
    color_map[agent] = [0.0, 0.0, 1.0]

    ax.imshow(color_map)

    # Draw grid lines (borders)
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(color='black', linestyle='-', linewidth=1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    placeholder.pyplot(fig)
    plt.close(fig)


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="Complete Pathfinder", layout="wide")

if "grid_size" not in st.session_state:
    st.session_state.grid_size = 10
    st.session_state.walls = np.zeros((10, 10), dtype=int)

if "dynamic_obstacles" not in st.session_state:
    st.session_state.dynamic_obstacles = set()

with st.sidebar:
    st.header("Settings")

    size = st.slider("Grid Size", 5, 25, 10)

    if size != st.session_state.grid_size:
        st.session_state.grid_size = size
        st.session_state.walls = np.zeros((size, size), dtype=int)
        st.session_state.dynamic_obstacles = set()

    algo = st.selectbox("Algorithm",
                        ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"])

    dls_limit = None
    if algo == "DLS":
        dls_limit = st.slider("Depth Limit", 1, 30, 10)

    speed = st.slider("Animation Speed", 0.01, 0.2, 0.05)

    st.divider()
    st.subheader("Dynamic Environment")
    enable_dynamic = st.checkbox("Enable Dynamic Hurdles", value=True)
    
    if enable_dynamic:
        dynamic_prob = st.slider("Dynamic Hurdle Probability (%)", 0, 100, 10)
        st.info("Dynamic hurdles may appear while the agent moves. The agent will detect and re-plan.")
    else:
        dynamic_prob = 0

    st.divider()

    if st.button("Clear Grid"):
        st.session_state.walls = np.zeros((size, size), dtype=int)
        st.session_state.dynamic_obstacles = set()
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

    wall_input = st.text_input("Enter wall coordinate (row,col)")

    colA, colB = st.columns(2)

    with colA:
        if st.button("Add Wall"):
            try:
                r, c = map(int, wall_input.split(","))
                if (r, c) != start and (r, c) != target:
                    st.session_state.walls[r, c] = 1
                    st.rerun()
            except:
                st.error("Invalid format")

    with colB:
        if st.button("Remove Wall"):
            try:
                r, c = map(int, wall_input.split(","))
                st.session_state.walls[r, c] = 0
                st.rerun()
            except:
                st.error("Invalid format")


with col2:
    placeholder = st.empty()
    status_placeholder = st.empty()
    
    draw_grid(st.session_state.walls, start, target, [], [], start, placeholder, st.session_state.dynamic_obstacles)

    if st.session_state.get("run", False):
        
        # Create a dynamic copy of the grid
        dynamic_grid = st.session_state.walls.copy()
        dynamic_obstacles = st.session_state.dynamic_obstacles.copy()
        
        # Initial pathfinding
        pf = Pathfinder(dynamic_grid, start, target)

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
            path_index = 0
            replans = 0
            
            while path_index < len(path):
                current_pos = path[path_index]
                
                # Check if we need to replan due to dynamic obstacles
                if path_index + 1 < len(path):
                    next_pos = path[path_index + 1]
                    
                    # Randomly spawn dynamic hurdle with given probability
                    if enable_dynamic and random.random() < (dynamic_prob / 100):
                        # Find empty cells ahead in the path (not start/target/walls)
                        potential_cells = []
                        for i in range(path_index + 1, min(path_index + 5, len(path))):
                            cell = path[i]
                            if cell != start and cell != target and dynamic_grid[cell] == 0:
                                potential_cells.append(cell)
                        
                        if potential_cells:
                            obstacle_cell = random.choice(potential_cells)
                            dynamic_obstacles.add(obstacle_cell)
                            dynamic_grid[obstacle_cell] = 1
                            
                            status_placeholder.warning(f"⚠️ Dynamic hurdle detected at {obstacle_cell}! Re-planning route... (Replan #{replans + 1})")
                            
                            # Replan from current position
                            pf = Pathfinder(dynamic_grid, current_pos, target)
                            
                            if algo == "BFS":
                                new_path, new_visited = pf.bfs()
                            elif algo == "DFS":
                                new_path, new_visited = pf.dfs()
                            elif algo == "UCS":
                                new_path, new_visited = pf.ucs()
                            elif algo == "DLS":
                                new_path, new_visited = pf.dls(dls_limit)
                            elif algo == "IDDFS":
                                new_path, new_visited = pf.iddfs()
                            else:
                                new_path, new_visited = pf.bidirectional()
                            
                            if not new_path:
                                st.error(f"No alternative path found after dynamic hurdle at {obstacle_cell}!")
                                break
                            
                            visited = visited | new_visited
                            path = new_path
                            path_index = 0
                            replans += 1
                            time.sleep(speed * 3)  # Pause to show re-planning
                            continue
                
                # Draw current state
                draw_grid(dynamic_grid, start, target, visited, path, current_pos, placeholder, dynamic_obstacles)
                time.sleep(speed)
                
                path_index += 1

            if path_index >= len(path):
                msg = f"✅ Reached Goal in {len(path)-1} steps."
                if replans > 0:
                    msg += f" Re-planned {replans} time(s) due to dynamic hurdles."
                status_placeholder.success(msg)

        st.session_state.run = False
        st.session_state.dynamic_obstacles = dynamic_obstacles
