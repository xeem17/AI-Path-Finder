import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg") # Prevents crashing on some systems
import matplotlib.pyplot as plt
import collections
import heapq
import time
import random

# ==========================================
# 1. SIMPLIFIED PATHFINDING LOGIC
# ==========================================

class Pathfinder:
    def __init__(self, grid, start, target):
        self.grid = grid
        self.start = start
        self.target = target
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # 8 Directions (Clockwise)
        self.directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

    def is_safe(self, r, c):
        # Check if inside the grid
        if r >= 0 and r < self.rows and c >= 0 and c < self.cols:
            # FIX: If it is 0, it is safe. Any other number (1, 2, 3...) is a Wall.
            if self.grid[r][c] == 0:
                return True
        return False

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

    # --- ALGORITHMS ---

    def bfs(self):
        queue = collections.deque([self.start])
        came_from = {self.start: None}
        visited = {self.start}

        while queue:
            current = queue.popleft()
            if current == self.target:
                yield visited, list(queue), self.build_path(came_from, current)
                return
            yield visited, list(queue), []
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

    def dfs(self):
        stack = [self.start]
        came_from = {self.start: None}
        visited = {self.start}

        while stack:
            current = stack.pop()
            if current == self.target:
                yield visited, stack, self.build_path(came_from, current)
                return
            yield visited, stack, []
            
            neighbors = self.get_neighbors(current)
            neighbors.reverse() # Reverse for correct clockwise order in stack
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append(neighbor)

    def ucs(self):
        pq = [(0, self.start)]
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}
        visited = set()

        while pq:
            current = heapq.heappop(pq)[1]
            visited.add(current)
            if current == self.target:
                yield visited, [x[1] for x in pq], self.build_path(came_from, current)
                return
            yield visited, [x[1] for x in pq], []
            
            for neighbor in self.get_neighbors(current):
                # Cost: 1.4 for diagonal, 1.0 for straight
                is_diag = (neighbor[0] != current[0] and neighbor[1] != current[1])
                move_cost = 1.414 if is_diag else 1.0
                new_cost = cost_so_far[current] + move_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
                    came_from[neighbor] = current

    def dls(self, limit):
        stack = [(self.start, 0)]
        came_from = {self.start: None}
        visited = {self.start}

        while stack:
            current, depth = stack.pop()
            if current == self.target:
                yield visited, [x[0] for x in stack], self.build_path(came_from, current), True
                return
            yield visited, [x[0] for x in stack], [], False
            
            if depth < limit:
                neighbors = self.get_neighbors(current)
                neighbors.reverse()
                for neighbor in neighbors:
                    if neighbor not in came_from:
                        came_from[neighbor] = current
                        visited.add(neighbor)
                        stack.append((neighbor, depth + 1))
        yield visited, [], [], False

    def iddfs(self):
        depth = 0
        while depth < 50: # Max depth safety
            for result in self.dls(depth):
                if len(result) == 4:
                    if result[3]: # Found
                        yield result[0], result[1], result[2]
                        return
                else:
                    yield result
            depth += 1

    def bidirectional(self):
        q_start = collections.deque([self.start])
        q_end = collections.deque([self.target])
        v_start = {self.start: None}
        v_end = {self.target: None}
        
        while q_start and q_end:
            # Expand Start
            if q_start:
                curr = q_start.popleft()
                for n in self.get_neighbors(curr):
                    if n not in v_start:
                        v_start[n] = curr
                        q_start.append(n)
                        if n in v_end:
                            p1 = self.build_path(v_start, n)
                            p2 = []
                            t = v_end[n]
                            while t: p2.append(t); t = v_end[t]
                            yield set(v_start)|set(v_end), list(q_start)+list(q_end), p1+p2
                            return
            # Expand End
            if q_end:
                curr = q_end.popleft()
                for n in self.get_neighbors(curr):
                    if n not in v_end:
                        v_end[n] = curr
                        q_end.append(n)
                        if n in v_start:
                            p1 = self.build_path(v_start, n)
                            p2 = []
                            t = v_end[n]
                            while t: p2.append(t); t = v_end[t]
                            yield set(v_start)|set(v_end), list(q_start)+list(q_end), p1+p2
                            return
            yield set(v_start)|set(v_end), list(q_start)+list(q_end), []

# ==========================================
# 2. VISUALIZATION
# ==========================================

def draw_grid(walls, start, end, visited, frontier, path, agent, placeholder):
    fig, ax = plt.subplots(figsize=(6, 6))
    rows, cols = walls.shape
    
    # 1. Base Map (White)
    color_map = np.zeros((rows, cols, 3)) + 1.0 
    
    # 2. Walls (Purple) - FIX: Any number > 0 is a wall
    color_map[walls > 0] = [0.5, 0.0, 0.5]

    # 3. Visited (Light Blue)
    for r, c in visited:
        if 0 <= r < rows and 0 <= c < cols: color_map[r, c] = [0.8, 0.9, 1.0]

    # 4. Frontier (Green)
    for r, c in frontier:
        if 0 <= r < rows and 0 <= c < cols: color_map[r, c] = [0.6, 1.0, 0.6]

    # 5. Path (Yellow)
    if path:
        for r, c in path: color_map[r, c] = [1.0, 0.8, 0.0]

    # 6. Start (Green) & End (Red)
    color_map[start[0], start[1]] = [0.0, 0.8, 0.0]
    color_map[end[0], end[1]] = [0.8, 0.0, 0.0]

    # 7. Agent (Blue)
    if agent: color_map[agent[0], agent[1]] = [0.0, 0.0, 1.0]

    ax.imshow(color_map)
    
    # Force Grid Lines (Graph Paper Look)
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black', linestyle='-', linewidth=1)
    
    placeholder.pyplot(fig)
    plt.close(fig)

# ==========================================
# 3. APP INTERFACE
# ==========================================

st.set_page_config(page_title="Pathfinder", layout="wide")

# Session State
if 'grid_size' not in st.session_state:
    st.session_state.grid_size = 10
if 'walls' not in st.session_state:
    st.session_state.walls = np.zeros((10, 10), dtype=int)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    size = st.slider("Grid Size", 5, 25, 10)
    if size != st.session_state.grid_size:
        st.session_state.grid_size = size
        st.session_state.walls = np.zeros((size, size), dtype=int)

    algo = st.selectbox("Algorithm", ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"])
    dls_lim = st.slider("DLS Limit", 1, 30, 10) if algo == "DLS" else 10
    speed = st.slider("Animation Speed", 0.01, 0.5, 0.05)
    
    st.subheader("Dynamic Obstacles")
    prob = st.slider("Spawn Probability", 0.0, 0.5, 0.10)

    if st.button("Reset Grid"):
        st.session_state.walls = np.zeros((size, size), dtype=int)
        st.rerun()
    
    st.markdown("---")
    # Using a run flag in session state ensures smooth simulation
    if st.button("Start Simulation", type="primary"):
        st.session_state.run = True

# --- MAIN LAYOUT ---
c1, c2 = st.columns([1, 1.5]) 

with c1:
    col_a, col_b = st.columns(2)
    sx = col_a.number_input("Start X", 0, size-1, 0)
    sy = col_b.number_input("Start Y", 0, size-1, 0)
    tx = col_a.number_input("Target X", 0, size-1, size-1)
    ty = col_b.number_input("Target Y", 0, size-1, size-1)
    
    st.write("Edit Grid (Type '1' or any number for Wall):")
    edited = st.data_editor(st.session_state.walls, height=400, use_container_width=True, key="editor")
    
    # Save updates from table
    if not np.array_equal(edited, st.session_state.walls):
        st.session_state.walls = edited
        st.rerun()

with c2:
    viz = st.empty()
    status = st.empty()
    
    start, target = (sx, sy), (tx, ty)
    
    # Draw Static Grid first
    draw_grid(st.session_state.walls, start, target, [], [], [], start, viz)

    if st.session_state.get('run', False):
        walls = st.session_state.walls.copy()
        agent = start
        steps = 0
        
        while agent != target:
            pf = Pathfinder(walls, agent, target)
            
            # Select Algo
            if algo == "BFS": gen = pf.bfs()
            elif algo == "DFS": gen = pf.dfs()
            elif algo == "UCS": gen = pf.ucs()
            elif algo == "Bidirectional": gen = pf.bidirectional()
            elif algo == "IDDFS": gen = pf.iddfs()
            elif algo == "DLS": 
                def wrap():
                    for i in pf.dls(dls_lim): 
                        if len(i)==4: yield i[0], i[1], i[2]
                        else: yield i
                gen = wrap()

            # Animate Plan
            path = []
            visited = set()
            for step in gen:
                visited, frontier, path = step[0], step[1], step[2]
                draw_grid(walls, start, target, visited, frontier, path, agent, viz)
                if path: break
            
            if not path:
                status.error("Stuck! No path found.")
                break
            
            # Move Agent
            if len(path) > 1: agent = path[1]; steps += 1
            
            # Random Obstacle
            if random.random() < prob:
                rx, ry = random.randint(0, size-1), random.randint(0, size-1)
                # Don't spawn on top of agent/start/target
                if (rx, ry) not in [agent, start, target]:
                    walls[rx, ry] = 1 # Mark as wall
                    status.warning(f"Obstacle spawned at {rx},{ry}")

            # Check Re-plan
            blocked = False
            for n in path[1:]:
                if walls[n[0], n[1]] > 0: # Check if blocked by any wall value
                    blocked = True; break
            
            if blocked:
                status.warning("Path blocked! Re-planning...")
                time.sleep(0.5)
            else:
                draw_grid(walls, start, target, visited, [], path, agent, viz)
                time.sleep(speed)

        if agent == target:
            status.success(f"Reached Goal in {steps} steps!")
            st.balloons()
            st.session_state.run = False
