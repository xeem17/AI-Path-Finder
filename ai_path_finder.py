import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import collections
import heapq
import time

# --- Part 1: Pathfinder Class (Logic) ---
class Pathfinder:
    def __init__(self, grid, start, target):
        self.grid = grid  # 0=Empty, 1=Wall
        self.start = start
        self.target = target
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def get_neighbors(self, node):
        r, c = node
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == 0:
                yield (nr, nc)

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(self.start)
        return path[::-1]

    # --- Algorithms (Generators for Animation) ---
    
    def bfs(self):
        queue = collections.deque([self.start])
        came_from = {self.start: None}
        visited = {self.start}
        
        while queue:
            current = queue.popleft()
            
            if current == self.target:
                yield visited, list(queue), self.reconstruct_path(came_from, current)
                return

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
            
            yield visited, list(queue), [] # Yield state for animation

    def dfs(self):
        stack = [self.start]
        came_from = {self.start: None}
        visited = {self.start}

        while stack:
            current = stack.pop()
            
            if current == self.target:
                yield visited, stack, self.reconstruct_path(came_from, current)
                return

            # Yield state before expanding neighbors
            yield visited, stack, []

            for neighbor in self.get_neighbors(current):
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
                yield visited, [n[1] for n in pq], self.reconstruct_path(came_from, current)
                return

            yield visited, [n[1] for n in pq], []

            for neighbor in self.get_neighbors(current):
                new_cost = current_cost + 1 # Uniform cost of 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(pq, (priority, neighbor))
                    came_from[neighbor] = current

    def bidirectional(self):
        # Two BFS searches
        q_start = collections.deque([self.start])
        q_end = collections.deque([self.target])
        
        visited_start = {self.start: None}
        visited_end = {self.target: None}
        
        while q_start and q_end:
            # Expand from start
            if q_start:
                curr_s = q_start.popleft()
                for neighbor in self.get_neighbors(curr_s):
                    if neighbor not in visited_start:
                        visited_start[neighbor] = curr_s
                        q_start.append(neighbor)
                        if neighbor in visited_end:
                            # Intersection found! Connect paths
                            path_a = self.reconstruct_path(visited_start, neighbor)
                            # Reconstruct backwards from end
                            path_b = []
                            curr = visited_end[neighbor]
                            while curr:
                                path_b.append(curr)
                                curr = visited_end[curr]
                            yield set(visited_start)|set(visited_end), list(q_start)+list(q_end), path_a + path_b
                            return
            
            # Expand from end
            if q_end:
                curr_e = q_end.popleft()
                for neighbor in self.get_neighbors(curr_e):
                    if neighbor not in visited_end:
                        visited_end[neighbor] = curr_e
                        q_end.append(neighbor)
                        if neighbor in visited_start:
                            # Intersection found
                            path_a = self.reconstruct_path(visited_start, neighbor)
                            path_b = []
                            curr = visited_end[neighbor]
                            while curr:
                                path_b.append(curr)
                                curr = visited_end[curr]
                            yield set(visited_start)|set(visited_end), list(q_start)+list(q_end), path_a + path_b
                            return

            yield set(visited_start)|set(visited_end), list(q_start)+list(q_end), []

    def dls(self, limit):
        # Helper for IDDFS
        stack = [(self.start, 0)] # Node, Depth
        came_from = {self.start: None}
        visited = {self.start} # For animation visualization only

        while stack:
            current, depth = stack.pop()
            
            if current == self.target:
                return visited, stack, self.reconstruct_path(came_from, current), True

            if depth < limit:
                for neighbor in self.get_neighbors(current):
                    if neighbor not in came_from: # simple cycle check
                        came_from[neighbor] = current
                        visited.add(neighbor)
                        stack.append((neighbor, depth + 1))
            
            yield visited, stack, [], False # Yielding False means "not found yet"
        
        return visited, [], [], False

    def iddfs(self):
        depth = 0
        while True:
            # Yield a clear signal that we are restarting with new depth
            dls_gen = self.dls(depth)
            found = False
            for step in dls_gen:
                if len(step) == 4: # Final return from DLS
                    visited, frontier, path, found = step
                    if found:
                        yield visited, frontier, path
                        return
                else:
                    yield step # Pass through animation frames
            
            depth += 1
            if depth > self.rows * self.cols: # Safety break
                return

# --- Part 2: Streamlit GUI ---

st.set_page_config(page_title="Pathfinding Visualizer", layout="wide")

# -- 1. State Management --
if 'grid_size' not in st.session_state:
    st.session_state.grid_size = 15
if 'walls' not in st.session_state:
    st.session_state.walls = np.zeros((15, 15), dtype=int)

# -- 2. Sidebar Controls --
st.sidebar.title("Configuration")

# Grid Settings
new_size = st.sidebar.slider("Grid Size", 5, 30, st.session_state.grid_size)
if new_size != st.session_state.grid_size:
    st.session_state.grid_size = new_size
    st.session_state.walls = np.zeros((new_size, new_size), dtype=int)

# Algorithm Selection
algo_choice = st.sidebar.selectbox(
    "Select Algorithm", 
    ["BFS", "DFS", "UCS", "Bidirectional", "IDDFS"]
)
speed = st.sidebar.slider("Animation Speed (sec)", 0.01, 0.5, 0.05)

# Dynamic Obstacles
st.sidebar.markdown("---")
prob = st.sidebar.slider("Random Wall Probability", 0.0, 0.5, 0.2)
if st.sidebar.button("Generate Random Walls"):
    st.session_state.walls = (np.random.rand(new_size, new_size) < prob).astype(int)
    # Ensure start/end are not walls
    st.session_state.walls[0, 0] = 0
    st.session_state.walls[new_size-1, new_size-1] = 0

if st.sidebar.button("Clear Walls"):
    st.session_state.walls = np.zeros((new_size, new_size), dtype=int)

# -- 3. Main Area: Setup --
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Positions")
    # Coordinates for Start/End (Simulating clicks which are hard in stateless web)
    c1, c2 = st.columns(2)
    start_x = c1.number_input("Start X", 0, new_size-1, 0)
    start_y = c2.number_input("Start Y", 0, new_size-1, 0)
    
    c3, c4 = st.columns(2)
    end_x = c3.number_input("Target X", 0, new_size-1, new_size-1)
    end_y = c4.number_input("Target Y", 0, new_size-1, new_size-1)
    
    start_pos = (start_x, start_y)
    end_pos = (end_x, end_y)

    st.subheader("Draw Walls")
    st.info("Edit the grid below to place/remove walls manually.")
    # Data Editor allows clicking cells to toggle values (0=Empty, 1=Wall)
    # We transpose for better visual layout in editor
    edited_walls = st.data_editor(
        st.session_state.walls, 
        key="wall_editor",
        use_container_width=True,
        height=300
    )
    # Sync back to session state
    if not np.array_equal(edited_walls, st.session_state.walls):
        st.session_state.walls = edited_walls
        st.rerun()

# -- 4. Visualization Helper --
def draw_grid(walls, start, end, visited, frontier, path, placeholder):
    """
    Draws the grid using Matplotlib and pushes it to the Streamlit placeholder.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    rows, cols = walls.shape
    
    # Base Map: 0 = White (Empty), 1 = Black (Wall)
    # We create a color map matrix
    color_map = np.zeros((rows, cols, 3)) + 1.0 # Default white
    
    # 1. Draw Walls (Black)
    for r in range(rows):
        for c in range(cols):
            if walls[r, c] == 1:
                color_map[r, c] = [0.2, 0.2, 0.2] # Dark Grey

    # 2. Draw Visited (Light Blue)
    for r, c in visited:
        color_map[r, c] = [0.6, 0.8, 1.0]

    # 3. Draw Frontier (Green)
    for r, c in frontier:
        color_map[r, c] = [0.6, 1.0, 0.6]

    # 4. Draw Path (Yellow)
    if path:
        for r, c in path:
            color_map[r, c] = [1.0, 0.8, 0.2]

    # 5. Draw Start (Blue) & End (Red)
    color_map[start[0], start[1]] = [0.0, 0.0, 1.0]
    color_map[end[0], end[1]] = [1.0, 0.0, 0.0]

    ax.imshow(color_map)
    
    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    
    placeholder.pyplot(fig)
    plt.close(fig)

# -- 5. Execution Logic --
with col2:
    st.subheader("Visualizer")
    viz_container = st.empty()
    stats_container = st.container()

    # Initial Draw
    draw_grid(st.session_state.walls, start_pos, end_pos, set(), [], [], viz_container)

    if st.button("Run Algorithm", type="primary"):
        # Validate positions
        if st.session_state.walls[start_pos] == 1 or st.session_state.walls[end_pos] == 1:
            st.error("Start or Target position is on a wall!")
        else:
            pf = Pathfinder(st.session_state.walls, start_pos, end_pos)
            
            # Select Generator
            algo_map = {
                "BFS": pf.bfs,
                "DFS": pf.dfs,
                "UCS": pf.ucs,
                "Bidirectional": pf.bidirectional,
                "IDDFS": pf.iddfs
            }
            generator = algo_map[algo_choice]()
            
            # Animation Loop
            visited_len = 0
            path_len = 0
            
            try:
                for visited, frontier, path in generator:
                    draw_grid(st.session_state.walls, start_pos, end_pos, visited, frontier, path, viz_container)
                    visited_len = len(visited)
                    time.sleep(speed)
                    if path:
                        path_len = len(path)
                        break
                
                # Final Stats
                with stats_container:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Path Length", path_len)
                    s2.metric("Nodes Explored", visited_len)
                    if algo_choice == "UCS":
                        s3.metric("Cost", path_len - 1 if path_len > 0 else 0)
                    
                    if not path:
                        st.error("No path found!")
                    else:
                        st.success("Target Reached!")
                        
            except Exception as e:
                st.error(f"Error executing algorithm: {e}")
