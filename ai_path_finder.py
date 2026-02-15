import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import collections
import heapq
import time
import random

# ==========================================
# 1. SIMPLIFIED PATHFINDING CLASS
# ==========================================

class Pathfinder:
    def __init__(self, grid, start, target):
        self.grid = grid
        self.start = start
        self.target = target
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # 8 Directions: Up, Up-Right, Right, Down-Right, Down, Down-Left, Left, Up-Left
        self.directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

    # Check if a cell is safe to walk on
    def is_safe(self, r, c):
        # Check if inside the grid
        if r >= 0 and r < self.rows:
            if c >= 0 and c < self.cols:
                # IMPORTANT: Only '1' is a wall. 
                # We can walk on 0 (Empty), 2 (Start), 3 (Target)
                if self.grid[r][c] != 1:
                    return True
        return False

    def get_neighbors(self, node):
        r = node[0]
        c = node[1]
        neighbors = []
        
        for direction in self.directions:
            dr = direction[0]
            dc = direction[1]
            
            new_r = r + dr
            new_c = c + dc
            
            if self.is_safe(new_r, new_c):
                neighbors.append((new_r, new_c))
                
        return neighbors

    def build_path(self, came_from, current):
        path = []
        while current is not None:
            path.append(current)
            # Go back one step
            current = came_from.get(current)
        
        # The path is backwards, so reverse it
        path.reverse()
        return path

    # --- SEARCH ALGORITHMS ---

    def bfs(self):
        # Queue for BFS
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

            # Show animation frame
            yield visited, list(queue), []
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

    def dfs(self):
        # Stack for DFS
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
            
            # Get neighbors and reverse them to keep Clockwise order
            neighbors = self.get_neighbors(current)
            neighbors.reverse()
            
            for neighbor in neighbors:
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
            current = item[1]
            
            visited.add(current)

            if current == self.target:
                # Helper to show frontier in animation
                frontier_list = []
                for x in pq:
                    frontier_list.append(x[1])
                    
                final_path = self.build_path(came_from, current)
                yield visited, frontier_list, final_path
                return

            frontier_list = []
            for x in pq:
                frontier_list.append(x[1])
            yield visited, frontier_list, []

            for neighbor in self.get_neighbors(current):
                # Calculate cost (1.4 for diagonal, 1.0 for straight)
                move_cost = 1.0
                if neighbor[0] != current[0] and neighbor[1] != current[1]:
                    move_cost = 1.414
                
                new_cost = cost_so_far[current] + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(pq, (priority, neighbor))
                    came_from[neighbor] = current

    def dls(self, limit):
        # Helper for IDDFS
        stack = []
        stack.append((self.start, 0)) # (Node, Depth)
        
        came_from = {}
        came_from[self.start] = None
        
        visited = set()
        visited.add(self.start)

        while len(stack) > 0:
            item = stack.pop()
            current = item[0]
            depth = item[1]

            # Helper for animation
            frontier_list = []
            for x in stack:
                frontier_list.append(x[0])

            if current == self.target:
                final_path = self.build_path(came_from, current)
                yield visited, frontier_list, final_path, True
                return

            yield visited, frontier_list, [], False

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
        max_depth = 50 
        
        while depth < max_depth:
            # Run DLS for this depth
            dls_generator = self.dls(depth)
            
            for result in dls_generator:
                if len(result) == 4:
                    visited = result[0]
                    frontier = result[1]
                    path = result[2]
                    found = result[3]
                    
                    if found == True:
                        yield visited, frontier, path
                        return
                else:
                    # Just an animation frame
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
                        
                        # Check if we met the other side
                        if neighbor in visited_end:
                            p1 = self.build_path(visited_start, neighbor)
                            
                            # Build p2 manually backwards
                            p2 = []
                            temp = visited_end[neighbor]
                            while temp is not None:
                                p2.append(temp)
                                temp = visited_end[temp]
                            
                            combined_path = p1 + p2
                            
                            # Combine visited sets for display
                            all_visited = set(visited_start.keys()) | set(visited_end.keys())
                            all_frontier = list(q_start) + list(q_end)
                            
                            yield all_visited, all_frontier, combined_path
                            return

            # 2. Expand from End
            if len(q_end) > 0:
                current_e = q_end.popleft()
                
                for neighbor in self.get_neighbors(current_e):
                    if neighbor not in visited_end:
                        visited_end[neighbor] = current_e
                        q_end.append(neighbor)
                        
                        if neighbor in visited_start:
                            p1 = self.build_path(visited_start, neighbor)
                            
                            p2 = []
                            temp = visited_end[neighbor]
                            while temp is not None:
                                p2.append(temp)
                                temp = visited_end[temp]
                            
                            combined_path = p1 + p2
                            
                            all_visited = set(visited_start.keys()) | set(visited_end.keys())
                            all_frontier = list(q_start) + list(q_end)
                            
                            yield all_visited, all_frontier, combined_path
                            return

            # Show animation frame
            all_visited = set(visited_start.keys()) | set(visited_end.keys())
            all_frontier = list(q_start) + list(q_end)
            yield all_visited, all_frontier, []

# ==========================================
# 2. VISUALIZATION HELPER
# ==========================================

def draw_grid(walls, start, end, visited, frontier, path, agent_pos, placeholder):
    # Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    rows = len(walls)
    cols = len(walls[0])
    
    # 1. Base Map (White)
    # Create a grid of colors, initialized to white [1, 1, 1]
    color_map = np.zeros((rows, cols, 3)) + 1.0 
    
    # 2. Draw Walls (Purple)
    for r in range(rows):
        for c in range(cols):
            if walls[r][c] == 1:
                color_map[r][c] = [0.5, 0.0, 0.5]

    # 3. Draw Visited Nodes (Light Blue)
    for node in visited:
        r = node[0]
        c = node[1]
        if r < rows and c < cols:
            color_map[r][c] = [0.8, 0.9, 1.0]

    # 4. Draw Frontier Nodes (Greenish)
    for node in frontier:
        r = node[0]
        c = node[1]
        if r < rows and c < cols:
            color_map[r][c] = [0.6, 1.0, 0.6]

    # 5. Draw Path (Yellow)
    if path is not None:
        for node in path:
            r = node[0]
            c = node[1]
            color_map[r][c] = [1.0, 0.8, 0.0]

    # 6. Draw Start (Green) & End (Red)
    color_map[start[0]][start[1]] = [0.0, 0.8, 0.0]
    color_map[end[0]][end[1]] = [0.8, 0.0, 0.0]

    # 7. Draw Agent (Blue Dot)
    if agent_pos is not None:
        r = agent_pos[0]
        c = agent_pos[1]
        color_map[r][c] = [0.0, 0.0, 1.0]

    # Show the image
    ax.imshow(color_map)
    
    # Clean up the chart (remove numbers on axes)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Send to Streamlit
    placeholder.pyplot(fig)
    plt.close(fig)

# ==========================================
# 3. MAIN APP (STREAMLIT)
# ==========================================

st.set_page_config(page_title="Pathfinder", layout="wide")

# -- Setup Session State --
if 'grid_size' not in st.session_state:
    st.session_state.grid_size = 15
if 'walls' not in st.session_state:
    # Initialize empty grid of 0s
    st.session_state.walls = np.zeros((15, 15), dtype=int)

# --- LEFT SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    
    # Grid Size
    new_size = st.slider("Grid Size", 5, 25, 15)
    if new_size != st.session_state.grid_size:
        st.session_state.grid_size = new_size
        st.session_state.walls = np.zeros((new_size, new_size), dtype=int)

    # Algorithm
    algo_choice = st.selectbox("Algorithm", ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"])
    
    dls_limit = 10
    if algo_choice == "DLS":
        dls_limit = st.slider("DLS Limit", 1, 30, 10)

    # Animation Speed
    speed = st.slider("Animation Speed (sec)", 0.01, 0.5, 0.05)

    # Dynamic Obstacles
    st.subheader("Dynamic Obstacles")
    spawn_prob = st.slider("Spawn Probability", 0.0, 0.5, 0.10)

    # Reset
    if st.button("Reset Grid"):
        st.session_state.walls = np.zeros((new_size, new_size), dtype=int)
        st.rerun()

    st.markdown("---")
    
    # Run Button Logic
    if st.button("Start Simulation", type="primary"):
        st.session_state.run_sim = True
    else:
        st.session_state.run_sim = False


# --- MAIN PAGE ---
col1, col2 = st.columns([1, 1.5]) 

# Left Column: Inputs & Grid Editor
with col1:
    c1, c2 = st.columns(2)
    with c1:
        sx = st.number_input("Start X", 0, new_size-1, 0)
        tx = st.number_input("Target X", 0, new_size-1, new_size-1)
    with c2:
        sy = st.number_input("Start Y", 0, new_size-1, 0)
        ty = st.number_input("Target Y", 0, new_size-1, new_size-1)

    start_pos = (sx, sy)
    target_pos = (tx, ty)

    st.write("Edit Grid (1 = Wall, 0 = Empty):")
    
    # Interactive Table
    edited_walls = st.data_editor(
        st.session_state.walls, 
        height=400, 
        use_container_width=True,
        key="editor"
    )
    
    # Save changes from table
    if not np.array_equal(edited_walls, st.session_state.walls):
        st.session_state.walls = edited_walls
        st.rerun()

# Right Column: Visualization
with col2:
    viz_placeholder = st.empty()
    status_text = st.empty()

    # Draw initial state
    draw_grid(st.session_state.walls, start_pos, target_pos, [], [], [], start_pos, viz_placeholder)

    # --- SIMULATION LOGIC ---
    if st.session_state.get('run_sim', False):
        
        # Make a copy so we don't ruin the original map permanently
        current_walls = st.session_state.walls.copy()
        agent_current = start_pos
        steps = 0
        
        # Loop until agent reaches target
        while agent_current != target_pos:
            
            # 1. Initialize Pathfinder
            pathfinder = Pathfinder(current_walls, agent_current, target_pos)
            
            # 2. Pick Algorithm
            algorithm_generator = None
            
            if algo_choice == "BFS":
                algorithm_generator = pathfinder.bfs()
            elif algo_choice == "DFS":
                algorithm_generator = pathfinder.dfs()
            elif algo_choice == "UCS":
                algorithm_generator = pathfinder.ucs()
            elif algo_choice == "Bidirectional":
                algorithm_generator = pathfinder.bidirectional()
            elif algo_choice == "IDDFS":
                algorithm_generator = pathfinder.iddfs()
            elif algo_choice == "DLS": 
                # Wrapper to make DLS match other algorithms
                def dls_wrapper():
                    for item in pathfinder.dls(dls_limit):
                        if len(item) == 4:
                            yield item[0], item[1], item[2]
                        else:
                            yield item
                algorithm_generator = dls_wrapper()

            # 3. Run Algorithm (Thinking Phase)
            path_found = []
            last_visited = set()
            
            for step in algorithm_generator:
                last_visited = step[0]
                frontier = step[1]
                path = step[2]
                
                # Update screen
                draw_grid(current_walls, start_pos, target_pos, last_visited, frontier, path, agent_current, viz_placeholder)
                
                if len(path) > 0:
                    path_found = path
                    break
            
            # If no path found, stop
            if len(path_found) == 0:
                status_text.error("No path found! Agent is stuck.")
                break
            
            # 4. Move Agent
            # path_found[0] is current, path_found[1] is next step
            if len(path_found) > 1:
                agent_current = path_found[1]
                steps = steps + 1
            
            # 5. Spawn Random Obstacle
            if random.random() < spawn_prob:
                rand_x = random.randint(0, new_size-1)
                rand_y = random.randint(0, new_size-1)
                
                # Don't spawn on Start, Target, or Agent
                safe_spots = [agent_current, start_pos, target_pos]
                if (rand_x, rand_y) not in safe_spots:
                    current_walls[rand_x][rand_y] = 1
                    status_text.warning(f"Obstacle spawned at {rand_x}, {rand_y}")

            # 6. Check if path is blocked
            is_blocked = False
            for node in path_found[1:]:
                r = node[0]
                c = node[1]
                if current_walls[r][c] == 1:
                    is_blocked = True
                    break
            
            if is_blocked:
                status_text.warning("Path blocked! Re-planning...")
                time.sleep(0.5)
            else:
                # Update screen and wait
                draw_grid(current_walls, start_pos, target_pos, last_visited, [], path_found, agent_current, viz_placeholder)
                time.sleep(speed)

        if agent_current == target_pos:
            status_text.success(f"Goal Reached in {steps} steps!")
            st.balloons()
            st.session_state.run_sim = False
