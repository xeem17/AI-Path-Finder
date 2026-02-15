import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg") # Optimizes plotting for web apps
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import collections
import heapq
import time
import random

# ==========================================
# 1. PATHFINDING LOGIC
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
        if r >= 0 and r < self.rows and c >= 0 and c < self.cols:
            # 0 is empty/safe. Any other number is a wall.
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
            neighbors.reverse()
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
                move_cost = 1.414 if (neighbor[0]!=current[0] and neighbor[1]!=current[1]) else 1.0
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
                        stack.append((neighbor, depth+1))
        yield visited, [], [], False

    def iddfs(self):
        depth = 0
        while depth < 50:
            for result in self.dls(depth):
                if len(result) == 4:
                    if result[3]: yield result[0], result[1], result[2]; return
                else: yield result
            depth += 1

    def bidirectional(self):
        q_start, q_end = collections.deque([self.start]), collections.deque([self.target])
        v_start, v_end = {self.start:None}, {self.target:None}
        while q_start and q_end:
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
    
    color_map = np.zeros((rows, cols, 3)) + 1.0 
    color_map[walls > 0] = [0.5, 0.0, 0.5] # Purple Walls
    
    for r, c in visited:
        if 0 <= r < rows and 0 <= c < cols: color_map[r, c] = [0.8, 0.9, 1.0] # Blue Visited
    for r, c in frontier:
        if 0 <= r < rows and 0 <= c < cols: color_map[r, c] = [0.6, 1.0, 0.6] # Green Frontier
    if path:
        for r, c in path: color_map[r, c] = [1.0, 0.8, 0.0] # Yellow Path

    color_map[start[0], start[1]] = [0.0, 0.8, 0.0] # Green Start
    color_map[end[0], end[1]] = [0.8, 0.0, 0.0]     # Red Target
    if agent: color_map[agent[0], agent[1]] = [0.0, 0.0, 1.0] # Blue Agent

    ax.imshow(color_map)
    
    # Force Grid Lines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black', linestyle='-', linewidth=1)
    
    placeholder.pyplot(fig)
    plt.close(fig)

def draw_editor_grid(walls, start, target):
    """Create interactive Plotly grid for editing walls"""
    rows, cols = walls.shape
    
    # Create display grid with text annotations
    display = np.zeros((rows, cols))
    text = [['' for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) == start:
                display[r, c] = 2
                text[r][c] = f'START<br>{r},{c}'
            elif (r, c) == target:
                display[r, c] = 3
                text[r][c] = f'TARGET<br>{r},{c}'
            elif walls[r, c] == 1:
                display[r, c] = 1
                text[r][c] = f'WALL<br>{r},{c}'
            else:
                display[r, c] = 0
                text[r][c] = f'{r},{c}'
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=display,
        text=text,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorscale=[
            [0, 'white'],      # Empty
            [0.33, 'purple'],  # Wall
            [0.66, 'green'],   # Start
            [1, 'red']         # Target
        ],
        showscale=False,
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Click to toggle<extra></extra>'
    ))
    
    fig.update_layout(
        width=500,
        height=500,
        xaxis=dict(side='top', dtick=1, showgrid=True, gridcolor='black'),
        yaxis=dict(dtick=1, showgrid=True, gridcolor='black', autorange='reversed'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ==========================================
# 3. STREAMLIT APP
# ==========================================

st.set_page_config(page_title="Pathfinder", layout="wide")

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
    
    st.subheader("Map Controls")
    c_rand, c_clear = st.columns(2)
    with c_rand:
        if st.button("Randomize"):
            st.session_state.walls = np.random.choice([0, 0, 0, 1], size=(size, size))
            st.rerun()
    with c_clear:
        if st.button("Clear Grid"):
            st.session_state.walls = np.zeros((size, size), dtype=int)
            st.rerun()
            
    st.subheader("Dynamic Obstacles")
    prob = st.slider("Spawn Probability", 0.0, 0.5, 0.10)
    
    st.markdown("---")
    if st.button("Start Simulation", type="primary"):
        st.session_state.run = True

# --- MAIN LAYOUT ---
c1, c2 = st.columns([1, 1.5]) 

with c1:
    col_a, col_b = st.columns(2)
    sx = col_a.number_input("Start X", 0, size-1, 0, key="start_x")
    sy = col_b.number_input("Start Y", 0, size-1, 0, key="start_y")
    tx = col_a.number_input("Target X", 0, size-1, size-1, key="target_x")
    ty = col_b.number_input("Target Y", 0, size-1, size-1, key="target_y")
    
    start, target = (sx, sy), (tx, ty)
    
    # Ensure start/target are safe
    st.session_state.walls[sx, sy] = 0
    st.session_state.walls[tx, ty] = 0
    
    st.write("**Add/Remove Walls:**")
    st.write("🔵 Start | 🔴 Target | 🟣 Wall | ⬜ Empty")
    
    wall_input = st.text_input("Enter position (row,col):", key="wall_input", placeholder="Example: 5,5")
    
    col_add, col_remove = st.columns(2)
    with col_add:
        if st.button("Add Wall", use_container_width=True, type="primary"):
            if not wall_input or ',' not in wall_input:
                st.error("Invalid format! Use: row,col (Example: 5,5)")
            else:
                try:
                    parts = wall_input.strip().split(',')
                    r, c = int(parts[0]), int(parts[1])
                    
                    if r < 0 or r >= size or c < 0 or c >= size:
                        st.error(f"Position out of bounds! Use 0-{size-1}")
                    elif (r, c) == start or (r, c) == target:
                        st.warning("Cannot place wall on start/target!")
                    else:
                        st.session_state.walls[r, c] = 1
                        st.success(f"✓ Wall added at ({r},{c})")
                        st.rerun()
                except (ValueError, IndexError):
                    st.error("Invalid format! Use numbers only: row,col")
    
    with col_remove:
        if st.button("Remove Wall", use_container_width=True):
            if not wall_input or ',' not in wall_input:
                st.error("Invalid format! Use: row,col (Example: 5,5)")
            else:
                try:
                    parts = wall_input.strip().split(',')
                    r, c = int(parts[0]), int(parts[1])
                    
                    if r < 0 or r >= size or c < 0 or c >= size:
                        st.error(f"Position out of bounds! Use 0-{size-1}")
                    else:
                        st.session_state.walls[r, c] = 0
                        st.success(f"✓ Wall removed at ({r},{c})")
                        st.rerun()
                except (ValueError, IndexError):
                    st.error("Invalid format! Use numbers only: row,col")
    
    st.divider()
    
    # Display current walls
    wall_positions = []
    for r in range(size):
        for c in range(size):
            if st.session_state.walls[r, c] == 1:
                wall_positions.append(f"({r},{c})")
    
    if wall_positions:
        st.write(f"**Current Walls ({len(wall_positions)}):**")
        st.text(", ".join(wall_positions[:20]))  # Show first 20
        if len(wall_positions) > 20:
            st.text(f"... and {len(wall_positions) - 20} more")
    else:
        st.info("No walls placed yet")

with c2:
    viz = st.empty()
    status = st.empty()
    
    # Draw Static Grid
    draw_grid(st.session_state.walls, start, target, [], [], [], start, viz)

    if st.session_state.get('run', False):
        walls = st.session_state.walls.copy()
        agent = start
        steps = 0
        
        while agent != target:
            pf = Pathfinder(walls, agent, target)
            
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

            path = []
            visited = set()
            for step in gen:
                visited, frontier, path = step[0], step[1], step[2]
                draw_grid(walls, start, target, visited, frontier, path, agent, viz)
                if path: break
            
            if not path:
                status.error("Stuck! No path found.")
                break
            
            if len(path) > 1: agent = path[1]; steps += 1
            
            if random.random() < prob:
                rx, ry = random.randint(0, size-1), random.randint(0, size-1)
                if (rx, ry) not in [agent, start, target]:
                    walls[rx, ry] = 1
                    status.warning(f"Obstacle spawned at {rx},{ry}")

            blocked = False
            for n in path[1:]:
                if walls[n[0], n[1]] > 0: blocked = True; break
            
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
