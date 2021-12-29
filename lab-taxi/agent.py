import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self,alpha=.1,gamma=.90,nA=6,eps=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
    
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state,i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.eps = 1/i_episode
        if np.random.random() < self.eps :
            return np.random.choice(self.nA)
        else:
            return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        current = self.Q[state][action]
        qsa_next = np.max(self.Q[next_state])
        target = reward + self.gamma * qsa_next
        self.Q[state][action] = current + self.alpha * (target - current)