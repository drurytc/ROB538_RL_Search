import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import copy
import visualization as vis
import agent as agt


class Evaluation:
    def __init__(self, environment, agent, agent1):
        self.env = environment
        self.start = environment.start
        self.global_reward = [0]  # Initialize global reward
        
        self.policy = agent.policy
        if agent1 is not None:
            self.policy1 = agent1.policy
        else:
            self.policy1 = agent1
        self.V = np.zeros((5, 10))  # Value function for agent 0
        self.V1 = np.zeros((5, 10))  # Value function for agent 1
        self.agent = agent

        self.alpha = 0.7
        self.epsilon = 0.8
        
    def evaluate(self, max_steps):
        i0, j0 = self.env.agents['0']
        
        steps = 0
        state_history = set()  # Track visited states for agent 0
        state_history1 = set()  # Track visited states for agent 1

        dones = 0
        while dones < 3 and steps < max_steps:
            # Take a step in the environment and receive the reward and done status
            done, reward = self.env.step(self.policy[i0 - 1][j0 - 1], 0)
             # Get the new state after taking the action
            i0_new, j0_new = self.env.agents['0']
             # Penalize revisiting the same state
            if (i0_new, j0_new) in state_history:
                reward += -3  # Penalty for revisiting
            # Add the current state to the history
            state_history.add((i0_new, j0_new))
            # Update value functions with shared global reward
            self.V[i0_new-1][j0_new-1] += self.alpha * (0*reward + (1*self.global_reward[0]) - self.V[i0_new - 1][j0_new - 1])
            steps += 1

            # Check if a target was reached and increment the global reward
            if done:
                dones += 1
                self.global_reward[0] += reward  # Increment the global reward
                self.policy[i0_new - 1][j0_new - 1] = random.choice([0, 1, 2, 3])
                if self.env.agents['0'] in self.env.targets:
                    self.env.targets.remove(self.env.agents['0'])

            if self.policy1 is not None:
                i1, j1 = self.env.agents['1']
                done1, reward1 = self.env.step(self.policy1[i1 - 1][j1 - 1], 1)
                i1_new, j1_new = self.env.agents['1']
                if (i1_new, j1_new) in state_history1:
                    reward1 += -3  # Penalty for revisiting
                state_history1.add((i1_new, j1_new))
                self.V1[i1_new-1][j1_new-1] += self.alpha * (1*reward1 + (0*self.global_reward[0]) - self.V1[i1_new - 1][j1_new - 1])

                if done1:
                    dones += 1
                    self.global_reward[0] += reward1
                    self.policy1[i1_new - 1][j1_new - 1] = random.choice([0, 1, 2, 3])
                    if self.env.agents['1'] in self.env.targets:
                        self.env.targets.remove(self.env.agents['1'])
           
                i1, j1 = i1_new, j1_new
            
            # Move to the next state
            i0, j0 = i0_new, j0_new
            

        # Reset the environment after the episode
        num_targets = len(self.env.targets)
        self.env.targets.clear()
        self.env.add_targets(num_targets)
        self.env.reset(0)
            
        if self.policy1 is not None:
            self.env.reset(1)
            return copy.deepcopy(self.V), copy.deepcopy(self.V1)
        return copy.deepcopy(self.V), None
    
    def iterate(self, max_steps, iterations):
        iteration = 0
        V0_0, V0_1 = self.evaluate(max_steps)
        
        while iteration < iterations:
            i, j = self.env.agents['0']
            # Store exploitative policies
            a0 = self.policy[i - 1][j - 1]

            a = random.choice([0, 1])
            # Update policies for exploration
            self.policy[i - 1][j - 1] = a

            # Evaluate the updated policies
            V1_0, V1_1 = self.evaluate(max_steps)

            th1 = (V1_0 - V0_0) > 0.7
            
            if th1.any() or random.random() < self.epsilon:
                V0_0 = copy.deepcopy(V1_0)  # Explore options
            else:
                self.policy[i - 1][j - 1] = a0  # Exploit Policy
            

            iteration += 1
            i, j = self.agent.get_next_state((i, j), self.policy[i - 1][j - 1])


            if self.policy1 is not None:
                i1, j1 = self.env.agents['1']
                b0 = self.policy1[i1 - 1][j1 - 1]
                b = random.choice([0, 1, 2, 3])
                self.policy1[i1 - 1][j1 - 1] = b
                th2 = (V1_1 - V0_1) > 0.7
                if th2.any() or random.random() < self.epsilon:
                    V0_1 = copy.deepcopy(V1_1)
                else:
                    self.policy1[i1 - 1][j1 - 1] = b0  

                i1, j1 = self.agent.get_next_state((i, j1), self.policy1[i1 - 1][j1 - 1])

        # Reset the environment after iterations
        self.env.reset(0)
        if self.policy1 is not None:
            self.env.reset(1)


def main():
    # Example usage
    env = vis.Environment(2, 2, (1, 10))
    agent = agt.Policy(env, 0)
    env.display()
    agent1 = agt.Policy(env, 1)

    # Reset the environment to starting positions
    env.reset(0)
    env.reset(1)
    policy_eval = Evaluation(env, agent, agent1)

    env.set_start((2, 3))
    env.reset(0)
    env.reset(1)

    policy_eval.iterate(20, iterations=1000)

    # Display the resulting policy and value function
    env.display()

    # Test the learned policy
    steps = 0
    steps1 = 0
    T1 = {'0': 0, '1': 0}
    T2 = {'0': 0, '1': 0}
    current_pos = env.agents['0']
    current_pos1 = env.agents['1']

    while steps < 20 or steps1 < 20:
        targets = {'T1': (4, 2)}
        done, reward = env.step(agent.policy[current_pos[0] - 1][current_pos[1] - 1], 0)
        current_pos = env.agents['0']
        steps += 1
        
        done1, reward1 = env.step(agent1.policy[current_pos1[0] - 1][current_pos1[1] - 1], 1)
        current_pos1 = env.agents['1']
        steps1 += 1

        if done:
            if current_pos == targets['T1']:
                T1['0'] += 1
                print(f"Agent 0 found Target T1 at {current_pos} in {steps} steps")
            if current_pos == targets['T2']:
                T2['0'] += 1
                print(f"Agent 0 found Target T2 at {current_pos} in {steps} steps")

        if done1:
            if current_pos1 == targets['T1']:
                T1['1'] += 1
                print(f"Agent 1 found Target T1 at {current_pos1} in {steps1} steps")
            if current_pos1 == targets['T2']:
                T2['1'] += 1
                print(f"Agent 1 found Target T2 at {current_pos1} in {steps1} steps")

    env.display()
    
if __name__ == '__main__':
    main()
