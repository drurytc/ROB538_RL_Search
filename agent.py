import numpy as np

class Policy:
    def __init__(self, environment, id=0):
        self.env = environment
        self.agent_id = str(id)
        self.direction = 2
        self.current_row = 1
        
        self.policy = np.full((5, 10), -1)
        
        self.val = np.full((5, 10), 0)
        self.policy = self.create_policy()
        print(self.policy)

    def create_policy(self):
        policy = [
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 1],
            [0, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ]
        return policy
        
    def get_next_state(self, state, action):
        x, y = state
        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1
        return x, y