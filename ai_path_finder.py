import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq
import time
import random

# --- Part 1: The Pathfinder Class ---
class Pathfinder:
    def __init__(self, grid, start, target):
        self.grid = grid
        self.start = start
        self.target = target
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # MOVEMENT ORDER: Clockwise starting Up (8 directions)
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

    # Helper to check if a cell is safe to visit
    def is_valid(self, r, c):
        if r >= 0 and r < self.rows:
            if c >= 0 and c < self.cols:
                if self.grid[r][c] == 0: # 0 means empty
                    return True
        return False

    def get_neighbors(self, node):
        r, c = node
        neighbors = []
        for direction in self.directions:
            dr = direction[0]
            dc = direction[1]
            nr = r + dr
            nc = c + dc
            
            if self.is_valid(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def build_path(self, came_from, current):
        path = []
        while current is not None:
            path.append(current)
            if current in came_from:
                current = came_from[current]
            else:
                current = None
        # Reverse path to go Start -> End
        path.reverse()
        return path

    # --- Search Algorithms ---

    def bfs(self):
        queue = collections.deque()
        queue.append(self.start)
        
        came_from = {}
        came_from[self.start] = None
        
        visited = set()
        visited.add(self.start)
        
        while len(queue) > 0:
            current = queue.popleft()
            
            if current == self.target:
                final_path = self.build_path(came_from, current)
                yield visited, list(queue), final_path
                return

            # Show current progress
            yield visited, list(queue), []

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

    def dfs(self):
        stack = []
        stack.append(self.start)
        
        came_from = {}
        came_from[self.start] = None
        
        visited = set()
        visited.add(self.start)

        while len(stack) > 0:
            current = stack.pop()
            
            if current == self.target:
                final_path = self.build_path(came_from, current)
                yield visited, stack, final_path
                return

            yield visited, stack, []

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append(neighbor)

    def ucs(self):
        # Priority Queue stores (cost, node)
        pq = []
        heapq.heappush(pq, (0, self.start))
        
        came_from = {}
        came_from[self.start] = None
        
        cost_so_far = {}
        cost_so_far[self.start] = 0
        
        visited = set()

        while len(pq) > 0:
            # Get node with lowest cost
            item = heapq.heappop(pq)
            current_cost = item[0]
            current = item[1]
            
            visited.add(current)

            if current == self.target:
                # Convert PQ to simple list for display
                frontier_list = []
                for x in pq:
                    frontier_list.append(x[1])
                    
                final_path = self.build_path(came_from, current)
                yield visited, frontier_list, final_path
                return

            # Display progress
            frontier_list = []
            for x in pq:
                frontier_list.append(x[1])
            yield visited, frontier_list, []

            for neighbor in self.get_neighbors(current):
                new_cost = current_cost + 1
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(pq, (priority, neighbor))
                    came_from[neighbor] = current

    def dls(self, limit):
        stack = []
        # Store (node, depth)
        stack.append((self.start, 0))
        
        came_from = {}
        came_from[self.start] = None
        
        visited = set()
        visited.add(self.start)

        while len(stack) > 0:
            item = stack.pop()
            current = item[0]
            depth = item[1]
            
            # Helper list for visualization
            frontier_list = []
            for x in stack:
                frontier_list.append(x[0])

            if current == self.target:
                final_path = self.build_path(came_from, current)
                yield visited, frontier_list, final_path, True
                return

            yield visited, frontier_list, [], False

            if depth < limit:
                for neighbor in self.get_neighbors(current):
                    if neighbor not in came_from:
                        came_from[neighbor] = current
                        visited.add(neighbor)
                        stack.append((neighbor, depth + 1))

    def iddfs(self):
        depth = 0
        max_depth = 50 # Avoid infinite loop
        
        while depth < max_depth:
            # Run DLS for current depth
            for result in self.dls(depth):
                # Unpack result
                if len(result) == 4:
                    visited = result[0]
                    frontier = result[1]
                    path = result[2]
                    found = result[3]
                    
                    if found == True:
                        yield visited, frontier, path
                        return
                else:
                    # Just showing animation
                    yield result
            
            depth = depth + 1

    def bidirectional(self):
        # Start side
        q_start = collections.deque()
        q_start.append(self.start)
        visited_start = {}
        visited_start[self.start] = None
        
        # End side
        q_end = collections.deque()
        q_end.append(self.target)
        visited_end = {}
        visited_end[self.target] = None
        
        while len(q_start) > 0 and len(q_end) > 0:
            
            # 1. Expand from Start
            if len(q_start) > 0:
                current_s = q_start.popleft()
                
                for neighbor in self.get_neighbors(current_s):
                    if neighbor not in visited_start:
                        visited_start[neighbor] = current_s
                        q_start.append(neighbor)
                        
                        # Check if paths meet
                        if neighbor in visited_end:
                            path_a = self.build_path(visited_start, neighbor)
                            
                            # Build path B manually (backwards)
                            path_b = []
                            curr = visited_end[neighbor]
                            while curr is not None:
                                path_b.append(curr)
                                curr = visited_end[curr]
                            
                            # Combine paths
                            full_path = path_a + path_b
                            
                            # Combine visited sets for display
                            all_visited = set(visited_start.keys()) | set(visited_end.keys())
                            all_frontier = list(q_start) + list(q_end)
                            
                            yield all_visited, all_frontier, full_path
                            return

            # 2. Expand from End
            if len(q_end) > 0:
                current_e = q_end.popleft()
                
                for neighbor in self.get_neighbors(current_e):
                    if neighbor not in visited_end:
                        visited_end[neighbor] = current_e
                        q_end.append(neighbor)
                        
                        if neighbor in visited_start:
                            path_a = self.build_path(visited_start, neighbor)
                            
                            path_b = []
                            curr = visited_end[neighbor]
                            while curr is not None:
                                path_b.append(curr)
                                curr = visited_end[curr]
                                
                            full_path = path_a + path_b
                            
                            all_visited = set(visited_start.keys()) | set(visited_end.keys())
                            all_frontier = list(q_start) + list(q_end)
                            
                            yield all_visited, all_frontier, full_path
                            return

            # Animation step
            all_visited = set(visited_start.keys()) | set(visited_end.keys())
            all_frontier = list(q_start) + list(q_end)
            yield all_visited, all_frontier, []


# --- Part 2: Streamlit GUI ---

st.set_page_config(page_title="AI Pathfinder", layout="wide")

# Initialize session state variables if they don't exist
if 'grid_size' not in st.session_state:
    st.session_state.grid_size = 15

if 'walls' not in st.session_state:
    # Create empty grid of 0s
    st.session_state.walls = np.zeros((15, 15), dtype=int)

# --- Sidebar Configuration ---
st.sidebar.title("Settings")

# 1. Grid Size
new_size = st.sidebar.slider("Grid Size", 5, 30, st.session_state.grid_size)
if new_size != st.session_state.grid_size:
    st.session_state.grid_size = new_size
    st.session_state.walls = np.zeros((new_size, new_size), dtype=int)

# 2. Algorithm
algo_choice = st.sidebar.selectbox("Select Algorithm", ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"])

dls_limit = 10
if algo_choice == "DLS":
    dls_limit = st.sidebar.slider("DLS Depth Limit", 1, 50, 10)

# 3. Speed & Obstacles
speed = st.sidebar.slider("Animation Speed", 0.01, 1.0, 0.05)
st.sidebar.markdown("---")
st.sidebar.write("Dynamic Obstacles")
spawn_prob = st.sidebar.slider("Spawn Probability", 0.0, 0.5, 0.1)

if st.sidebar.button("Clear Grid"):
    st.session_state.walls = np.zeros((new_size, new_size), dtype=int)


# --- Main Page Layout ---
col1, col2 = st.columns([1, 2])

# Left Column: Setup
with col1:
    st.subheader("Map Setup")
    
    c1, c2 = st.columns(2)
    start_x = c1.number_input("Start Row", 0, new_size-1, 0)
    start_y = c2.number_input("Start Col", 0, new_size-1, 0)
    
    c3, c4 = st.columns(2)
    end_x = c3.number_input("Target Row", 0, new_size-1, new_size-1)
    end_y = c4.number_input("Target Col", 0, new_size-1, new_size-1)
    
    start_pos = (start_x, start_y)
    end_pos = (end_x, end_y)

    st.write("Click cells to place walls (0=Empty, 1=Wall):")
    # Interactive table editor
    edited_walls = st.data_editor(st.session_state.walls, key="editor", height=300)
    
    # Save changes
    if not np.array_equal(edited_walls, st.session_state.walls):
        st.session_state.walls = edited_walls
        st.rerun()

# Helper function to draw the grid
def draw_grid(walls, start, end, visited, frontier, path, agent_pos, placeholder):
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    rows = len(walls)
    cols = len(walls[0])
    
    # Color Scheme
    # 0 = White (Empty)
    # 1 = Black (Wall)
    grid_colors = np.zeros((rows, cols, 3)) + 1.0 

    # Color Walls
    for r in range(rows):
        for c in range(cols):
            if walls[r][c] == 1:
                grid_colors[r][c] = [0.2, 0.2, 0.2] # Dark Grey

    # Color Visited Nodes (Blueish)
    for node in visited:
        r, c = node
        grid_colors[r][c] = [0.6, 0.8, 1.0]

    # Color Frontier (Greenish)
    for node in frontier:
        r, c = node
        grid_colors[r][c] = [0.6, 1.0, 0.6]

    # Color Path (Yellow)
    if path is not None:
        for node in path:
            r, c = node
            grid_colors[r][c] = [1.0, 0.8, 0.2]

    # Start (Blue) & End (Red)
    grid_colors[start[0]][start[1]] = [0.0, 0.0, 1.0]
    grid_colors[end[0]][end[1]] = [1.0, 0.0, 0.0]
    
    # Agent Position (Purple)
    if agent_pos is not None:
        grid_colors[agent_pos[0]][agent_pos[1]] = [0.5, 0.0, 0.5]

    ax.imshow(grid_colors)
    
    # Hide axis numbers
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw it on Streamlit
    placeholder.pyplot(fig)
    plt.close(fig)


# Right Column: Execution
with col2:
    st.subheader("Simulation")
    
    # Placeholders for dynamic updates
    viz_area = st.empty()
    status_msg = st.empty()
    
    # Draw initial state
    draw_grid(st.session_state.walls, start_pos, end_pos, [], [], [], start_pos, viz_area)

    if st.button("Start Simulation", type="primary"):
        
        # Copy walls so we don't mess up the original map
        current_walls = st.session_state.walls.copy()
        agent_curr = start_pos
        
        step_count = 0
        
        # Main Agent Loop
        while agent_curr != end_pos:
            status_msg.info(f"Agent at {agent_curr}. Planning...")
            
            # 1. Initialize Pathfinder
            pf = Pathfinder(current_walls, agent_curr, end_pos)
            
            # 2. Select Algorithm
            generator = None
            if algo_choice == "BFS":
                generator = pf.bfs()
            elif algo_choice == "DFS":
                generator = pf.dfs()
            elif algo_choice == "UCS":
                generator = pf.ucs()
            elif algo_choice == "Bidirectional":
                generator = pf.bidirectional()
            elif algo_choice == "IDDFS":
                generator = pf.iddfs()
            elif algo_choice == "DLS":
                # Wrapper for DLS to match format
                def dls_wrap():
                    for res in pf.dls(dls_limit):
                        if len(res) == 4:
                            # It's the final result
                            yield res[0], res[1], res[2]
                        else:
                            yield res
                generator = dls_wrap()

            # 3. Run Search (Thinking Process)
            found_path = []
            visited_display = set()
            
            for step in generator:
                visited_display = step[0]
                frontier_display = step[1]
                path_display = step[2]
                
                # Show animation
                draw_grid(current_walls, start_pos, end_pos, visited_display, frontier_display, path_display, agent_curr, viz_area)
                
                if len(path_display) > 0:
                    found_path = path_display
                    break
            
            # Check if stuck
            if len(found_path) == 0:
                status_msg.error("No path found! Agent is stuck.")
                break

            # 4. Move Agent (Take 1 step)
            # found_path[0] is current, found_path[1] is next
            if len(found_path) > 1:
                next_step = found_path[1]
                agent_curr = next_step
                step_count += 1
            
            # 5. Dynamic Obstacle Event
            rand_val = random.random()
            if rand_val < spawn_prob:
                # Pick random spot
                rx = random.randint(0, new_size - 1)
                ry = random.randint(0, new_size - 1)
                
                # Safety check: Don't spawn on Start, End, or Agent
                safe_spots = [start_pos, end_pos, agent_curr]
                if (rx, ry) not in safe_spots:
                    current_walls[rx][ry] = 1 # Add wall
                    status_msg.warning(f"Obstacle appeared at {(rx, ry)}!")

            # 6. Check if path is blocked
            is_blocked = False
            # Check remaining steps in path
            for i in range(1, len(found_path)):
                p_node = found_path[i]
                pr, pc = p_node
                if current_walls[pr][pc] == 1:
                    is_blocked = True
                    break
            
            if is_blocked:
                status_msg.warning("Path blocked! Re-calculating...")
                time.sleep(0.5)
            else:
                # Update visual and wait
                draw_grid(current_walls, start_pos, end_pos, visited_display, [], found_path, agent_curr, viz_area)
                time.sleep(speed)

        if agent_curr == end_pos:
            status_msg.success(f"Goal Reached in {step_count} steps!")
            st.balloons()
