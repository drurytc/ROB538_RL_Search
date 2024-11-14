import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import copy

class Environment:
    def __init__(self, num_agents=2, num_targets=1, loc=(1, 10)):
        self.grid = np.zeros((7, 12))
        self.start = loc
        self.agents = dict()
        self.targets = list()
        self.agent_paths = {str(i): [self.start] for i in range(num_agents)}  # Track paths for each agent
        self.agent_colors = ['blue', 'green'] 
        self.add_category()
        self.add_targets(num_targets)
        self.add_agent(num_agents, self.start)

    def add_category(self):
        for i in range(np.shape(self.grid)[0]):
            for j in range(np.shape(self.grid)[1]):
                if (i == 0 or j == 0) or (i == 6 or j == 11):
                    self.grid[i][j] = 1
                else:
                    self.grid[i][j] = 0

    def add_targets(self, num):
        if num == 1:
            self.targets.append((4, 2))
            self.grid[4][2] = -1
        if num == 2:
            self.targets.append((4, 2))
            self.targets.append((1, 9))
            self.grid[4][2] = -1
            self.grid[1][9] = -1

    def add_agent(self, num, loc):
        for i in range(num):
            self.agents[str(i)] = loc

    def set_start(self, loc):
        self.start = loc

    def display(self):
        cmap = ListedColormap(['red', 'white', 'black'])
        img = plt.imshow(self.grid, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
        cbar = plt.colorbar(img, ticks=[-1, 0, 1])
        cbar.set_label('Components')
        cbar.ax.set_yticklabels(['Target', 'Floor', 'Wall'])
        plt.title('Environment with Agent Paths')
        plt.gca().invert_yaxis()
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        ax = plt.gca()

        # Plot agent paths with lines connecting each step, using different colors for each agent
        for agent_id, path in self.agent_paths.items():
            color = self.agent_colors[int(agent_id) % len(self.agent_colors)]  # Cycle through colors if more than available
            # Unzip the path into x (columns) and y (rows)
            y_positions, x_positions = zip(*path)
            # Plot the path with lines
            ax.plot(x_positions, y_positions, color=color, linestyle='--', linewidth=2, marker='o', markersize=5, alpha=0.5)
        
        # Mark the final position of each agent
        for agent_id, pos in self.agents.items():
            color = self.agent_colors[int(agent_id) % len(self.agent_colors)]
            ax.scatter(pos[1], pos[0], color=color, s=100, label=f'Agent {agent_id}')

        ax.set_xticks(np.arange(-0.5, self.grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)

        # Only show the legend once to avoid duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        plt.show()

    def reset(self, id):
        self.agents[str(id)] = self.start
        num = len(self.targets)
        self.targets.clear()
        self.add_targets(num)
        self.agent_paths[str(id)] = [self.start]  # Reset the path to the starting position
        
    def step(self, action, id):
        id = str(id)
        current_pos = self.agents[id]
        new_pos = list(current_pos)
        
        # Update position based on action
        if action == 1 and new_pos[0] < 5:  # Move down
            new_pos[0] += 1
        elif action == 0 and new_pos[0] > 1:  # Move up
            new_pos[0] -= 1
        elif action == 2 and new_pos[1] > 1:  # Move left
            new_pos[1] -= 1
        elif action == 3 and new_pos[1] < 10:  # Move right
            new_pos[1] += 1

        # Update the agent's position and path
        self.agents[id] = tuple(new_pos)
        self.agent_paths[id].append(tuple(new_pos))

        # Check if the agent reached a target
        if tuple(new_pos) in self.targets:
            self.targets.remove(tuple(new_pos))
            return True, 20  # Reached a target
        else:
            return False, -1  # Not a target
        
    