AI Pathfinder
1. Introduction

AI Pathfinder is a Python-based project designed to demonstrate and analyze core pathfinding algorithms on a 2D grid. The system allows users to interactively explore algorithmic behavior, visualize the search process, and measure performance metrics such as path length, nodes explored, and path cost.

The project is ideal for educational purposes, algorithm analysis, or integration into larger AI systems.

2. Supported Algorithms

The following uninformed search algorithms are implemented:

Breadth-First Search (BFS) – Explores the shallowest nodes first.

Depth-First Search (DFS) – Explores the deepest nodes first using a stack-based approach.

Uniform-Cost Search (UCS) – Expands the node with the lowest cumulative cost.

Depth-Limited Search (DLS) – DFS with a fixed depth limit.

Iterative Deepening DFS (IDDFS) – Repeated DLS with increasing depth until the goal is found.

Bidirectional Search – Simultaneously searches forward from the start and backward from the target.

3. Features

Configurable grid size (default: 30×30)

Interactive placement of start and target nodes

Add and remove walls

Enable dynamic obstacles with adjustable probability

Real-time visualization of:

Frontier cells

Explored cells

Final path

Display algorithm metrics:

Path length

Nodes explored

Path cost (for UCS)

4. Requirements

Python 3.8 or higher

Install dependencies via pip:

pip install streamlit


Optional (if using Pygame GUI):

pip install pygame

5. Usage Instructions
5.1 Command-Line Execution (No GUI)

To test all algorithms on a predefined grid:

python pathfinder.py


Expected outputs include:

Path found (length)

Number of nodes explored

Path cost (for UCS)

5.2 Streamlit Interactive GUI

To run the interactive GUI:

streamlit run gui.py


GUI Features:

Adjust grid size using a slider

Click on cells to set start, target, and walls

Toggle dynamic obstacles and configure probability

Select an algorithm from the dropdown menu

Click Run to visualize the search algorithm step by step

Reset buttons are available to clear search results or the entire grid

5.3 Optional Pygame GUI

For an interactive Pygame-based GUI:

python pygame_gui.py


This GUI supports similar interactive features with real-time visualization.

7. Notes and Recommendations

Diagonal movement is supported by default.

Dynamic obstacles may appear randomly during searches; disabling them provides deterministic behavior.

Use the clear/reset functions to reset walls, start, and target nodes as needed.
