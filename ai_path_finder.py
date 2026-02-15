import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq
import time
import random

# ==========================================
# 1. CORE PATHFINDING LOGIC (With Generators)
# ==========================================

class Pathfinder:
    def __init__(self, grid, start, target):
        self.grid = grid
        self.start = start
        self.target = target
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # 8 Directions (Clockwise starting Up)
        self.directions = [
            (-1, 0),  # Up
            (-1, 1),  # Up-Right
            (0, 1),   # Right
            (1, 1),   # Down-Right
            (1, 0),   # Down
            (1, -1),  # Down-Left
            (0, -1),  # Left
            (-1, -1)  # Up-Left
        ]

    def is_valid(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.grid[r][c] == 0:  # 0 is empty
                return True
        return False

    def get_neighbors(self, node):
        r, c = node
        neighbors = []
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def reconstruct_path(self, came_from, current):
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return path[::-1]  # Reverse to get Start -> End

    # --- ALGORITHMS (Generators for Animation) ---

    def bfs(self):
        queue = collections.deque([self.start])
        came_from = {self.start: None}
        visited = {self.start}

        while queue:
            current = queue.popleft()
            if current == self.target:
                yield visited, list(queue), self.reconstruct_path(came_from, current)
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
                yield visited, stack, self.reconstruct_path(came_from, current)
                return

            yield visited, stack, []
            
            # Reverse neighbors to preserve clockwise order when popping
            neighbors = self.get_neighbors(current)
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append(neighbor)

    def ucs(self):
        # Priority Queue: (cost, node)
        pq = [(0, self.start)]
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}
        visited = set()

        while pq:
            current_cost, current = heapq.heappop(pq)
            visited.add(current)

            if current == self.target:
                # Extract nodes from PQ for visualization
                frontier = [node for _, node in pq]
                yield visited, frontier, self.reconstruct_path(came_from, current)
                return

            frontier = [node for _, node in pq]
            yield visited, frontier, []

            for neighbor in self.get_neighbors(current):
                # Cost: 1.414 for diagonal, 1 for straight
                is_diag = (neighbor[0] != current[0] and neighbor[1] != current[1])
                move_cost = 1.414 if is_diag else 1.0
                new_cost = cost_so_far[current] + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
                    came_from[neighbor] = current

    def dls(self, limit):
        stack = [(self.start, 0)] # Node, Depth
        came_from = {self.start: None}
        visited = {self.start}

        while stack:
            current, depth = stack.pop()

            if current == self.target:
                frontier = [n[0] for n in stack]
                yield visited, frontier, self.reconstruct_path(came_from, current), True
                return

            frontier = [n[0] for n in stack]
            yield visited, frontier, [], False

            if depth < limit:
                neighbors = self.get_neighbors(current)
                for neighbor in reversed(neighbors):
                    if neighbor not in came_from: # Simple cycle prevention
                        came_from[neighbor] = current
                        visited.add(neighbor)
                        stack.append((neighbor, depth + 1))
        
        # Not found at this limit
        yield visited, [], [], False

    def iddfs(self):
        depth = 0
        max_depth = 50 # Safety limit
        while depth < max_depth:
            # Call DLS generator
            dls_gen = self.dls(depth)
            for result in dls_gen:
                if len(result) == 4: # Standard DLS yield
                    visited, frontier, path, found = result
                    if found:
                        yield visited, frontier, path
                        return
                else:
                    yield result
            depth += 1

    def bidirectional(self):
        q_start = collections.deque([self.start])
        q_end = collections.deque([self.target])
        
        visited_start = {self.start: None}
        visited_end = {self.target: None}
        
        while q_start and q_end:
            # 1. Expand Start
            if q_start:
                curr = q_start.popleft()
                for neighbor in self.get_neighbors(curr):
                    if neighbor not in visited_start:
                        visited_start[neighbor] = curr
                        q_start.append(neighbor)
                        if neighbor in visited_end:
                            # Intersection!
                            p1 = self.reconstruct_path(visited_start, neighbor)
                            
                            # Reconstruct P2 manually backwards
                            p2 = []
                            temp = visited_end[neighbor]
                            while temp:
                                p2.append(temp)
                                temp = visited_end[temp]
                                
                            yield set(visited_start)|set(visited_end), list(q_start)+list(q_end), p1 + p2
                            return

            # 2. Expand End
            if q_end:
                curr = q_end.popleft()
                for neighbor in self.get_neighbors(curr):
                    if neighbor not in visited_end:
                        visited_end[neighbor] = curr
                        q_end.append(neighbor)
                        if neighbor in visited_start:
                            # Intersection!
                            p1 = self.reconstruct_path(visited_start, neighbor)
                            p2 = []
                            temp = visited_end[neighbor]
                            while temp:
                                p2.append(temp)
                                temp = visited_end[temp]
                            yield set(visited_start)|set(visited_end), list(q_start)+list(q_end), p1 + p2
                            return

            yield set(visited_start)|set(visited_end), list(q_start)+list(q_end), []


# ==========================================
# 2. VISUALIZATION HELPER
# ==========================================

def render_grid(walls, start, end, visited, frontier, path, agent_pos, placeholder):
    fig, ax = plt.subplots(figsize=(4, 4))
    rows, cols = walls.shape
    
    # Base Color: White
    color_map = np.zeros((rows, cols, 3)) + 1.0 
    
    # Walls: Black
    color_map[walls == 1] = [0, 0, 0]

    # Visited: Light Blue
    for r, c in visited:
        color_map[r, c] = [0.6, 0.8, 1.0]

    # Frontier: Greenish
    for r, c in frontier:
        color_map[r, c] = [0.6, 1.0, 0.6]

    # Path: Gold/Yellow
    if path:
        for r, c in path:
            color_map[r, c] = [1.0, 0.8, 0.0]

    # Start: Green, End: Red
    color_map[start[0], start[1]] = [0, 1, 0]
    color_map[end[0], end[1]] = [1, 0, 0]

    # Agent: Purple
    if agent_pos:
        color_map[agent_pos[0], agent_pos[1]] = [0.5, 0, 0.5]

    ax.imshow(color_map)
    ax.set_xticks([]); ax.set_yticks([]) # Hide ticks
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    placeholder.pyplot(fig)
    plt.close(fig)


# ==========================================
# 3. STREAMLIT GUI
# ==========================================

st.set_page_config(page_title="AI Pathfinder (Streamlit)", layout="wide")

# --- Session State Init ---
if 'grid_size' not in st.session_state:
    st.session_state.grid_size = 15
if 'walls' not in st.session_state:
    st.session_state.walls = np.zeros((15, 15), dtype=int)

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# Grid Size
size = st.sidebar.slider("Grid Size", 5, 25, 15)
if size != st.session_state.grid_size:
    st.session_state.grid_size = size
    st.session_state.walls = np.zeros((size, size), dtype=int)

# Algorithm Selection
algo_name = st.sidebar.selectbox("Algorithm", ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"])
dls_lim = 10
if algo_name == "DLS":
    dls_lim = st.sidebar.slider("DLS Depth Limit", 1, 30, 10)

# Settings
speed = st.sidebar.slider("Animation Speed (sec)", 0.01, 1.0, 0.05)
st.sidebar.markdown("---")
st.sidebar.subheader("Dynamic Obstacles")
obs_prob = st.sidebar.slider("Spawn Probability", 0.0, 0.5, 0.1)

if st.sidebar.button("Reset Grid"):
    st.session_state.walls = np.zeros((size, size), dtype=int)
    st.rerun()

# --- Main Layout ---
col_map, col_viz = st.columns([1, 2])

with col_map:
    st.subheader("Map Setup")
    c1, c2 = st.columns(2)
    sx = c1.number_input("Start X", 0, size-1, 0)
    sy = c2.number_input("Start Y", 0, size-1, 0)
    tx = c1.number_input("Target X", 0, size-1, size-1)
    ty = c2.number_input("Target Y", 0, size-1, size-1)
    
    start_pos = (sx, sy)
    target_pos = (tx, ty)

    st.write("Edit Grid (1 = Wall, 0 = Empty):")
    # Streamlit Data Editor for interactive grid
    edited = st.data_editor(st.session_state.walls, height=300, key="grid_editor")
    
    # Sync edits
    if not np.array_equal(edited, st.session_state.walls):
        st.session_state.walls = edited
        st.rerun()

with col_viz:
    st.subheader("Live Simulation")
    viz_box = st.empty()
    status = st.empty()

    # Initial Render
    render_grid(st.session_state.walls, start_pos, target_pos, [], [], [], start_pos, viz_box)

    if st.button("Start Simulation", type="primary"):
        
        # Working variables
        current_walls = st.session_state.walls.copy()
        agent_loc = start_pos
        steps = 0
        replans = 0
        
        # --- AGENT LOOP ---
        while agent_loc != target_pos:
            status.info(f"Step {steps}: Planning path from {agent_loc}...")
            
            # 1. PLAN
            pf = Pathfinder(current_walls, agent_loc, target_pos)
            
            # Select Generator
            gen = None
            if algo_name == "BFS": gen = pf.bfs()
            elif algo_name == "DFS": gen = pf.dfs()
            elif algo_name == "UCS": gen = pf.ucs()
            elif algo_name == "Bidirectional": gen = pf.bidirectional()
            elif algo_name == "IDDFS": gen = pf.iddfs()
            elif algo_name == "DLS":
                # Wrap DLS to match standard yield format
                def dls_wrapper():
                    for item in pf.dls(dls_lim):
                        if len(item) == 4: yield item[0], item[1], item[2]
                        else: yield item
                gen = dls_wrapper()

            # Run Animation (Thinking Process)
            path_found = []
            final_visited = set()
            
            for visited, frontier, path in gen:
                final_visited = visited
                render_grid(current_walls, start_pos, target_pos, visited, frontier, path, agent_loc, viz_box)
                if path:
                    path_found = path
                    break
            
            if not path_found:
                status.error("Agent Stuck! No path found.")
                break

            # 2. MOVE (Agent takes 1 step)
            # path_found[0] is current location, [1] is next step
            if len(path_found) > 1:
                next_step = path_found[1]
                agent_loc = next_step
                steps += 1
            
            # 3. DYNAMIC OBSTACLE EVENT
            if random.random() < obs_prob:
                # Spawn random wall
                rx = random.randint(0, size-1)
                ry = random.randint(0, size-1)
                # Don't spawn on agent, start, or target
                if (rx, ry) not in [agent_loc, start_pos, target_pos]:
                    current_walls[rx, ry] = 1
                    status.warning(f"Dynamic Obstacle spawned at {(rx, ry)}!")

            # 4. CHECK IF BLOCKED
            # Check if the NEXT step in our calculated path is now a wall
            is_blocked = False
            for node in path_found[1:]:
                if current_walls[node[0], node[1]] == 1:
                    is_blocked = True
                    break
            
            if is_blocked:
                status.warning("Path blocked! Re-calculating...")
                replans += 1
                time.sleep(0.5)
            else:
                # Just wait slightly to show movement
                render_grid(current_walls, start_pos, target_pos, final_visited, [], path_found, agent_loc, viz_box)
                time.sleep(speed)

        if agent_loc == target_pos:
            status.success(f"Target Reached! Total Steps: {steps}, Re-plans: {replans}")
            st.balloons()
