import preprocessing as prep
from constant_features import *
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

class DOTA2_surrogate:
    def __init__(self, model, scaler):
        self.state = None
        self.agent = None
        self.model = model
        self.scaler = scaler
        self.step_nb = 0

    def reset(self):
        # Reset env
        self.state = np.array(pd.read_csv("Surrogate/Init_state.csv", index_col=0, names=['init'])['init'])
        self.step_nb = 0
        print("Env Reset")
        return self.state

    def step(self, action):
        if self.state is None:
            self.reset()

        # Remove constants
        if prep.remove_constant_features:
            state_transformed = remove_constants(self.state)
        else:
            state_transformed = self.state

        # Transform state : remove features and scale
        state_transformed = self.scaler.transform(state_transformed.reshape(1, -1))

        # Transform action into vector
        action_vect = prep.actions_to_vect(action).reshape(1, -1)

        # Create input for surrogate
        x = np.concatenate([state_transformed, action_vect], axis=1)

        # Predict next state
        y = self.model.predict(x)

        # Revert scaling
        y = self.scaler.inverse_transform(y)[0]

        # Add constant features back
        if prep.remove_constant_features:
            y = add_constants(y)

        self.state = y

        # Increment step count
        self.step_nb += 1
        return self.state

    def render(self, mode='human'):
        t = self.state[56]
        x, y = self.state[26:28]
        print("%d > Time: %d | (%d , %d)" % (self.step_nb, t, x, y))
        return self.state

    def play_game(self, decision, nb_steps):
        # Run the surrogate with a random agent
        self.reset()  # Initial state

        t, pos_x, pos_y = [], [], []  # Time and position to plot

        for i in range(nb_steps):  # 100 steps
            # Choose a random action
            a = decision(self.state)

            # Compute next step
            self.step(a)

            # To print details, uncomment:
            self.render()

            # Store data
            t.append(self.state[56])
            pos_x.append(self.state[26])
            pos_y.append(self.state[27])

        # Plot data
        plt.plot(t)
        plt.xlabel("Steps")
        plt.ylabel("Game time")
        plt.show()

        plt.figure(figsize=(9, 8))
        sc = plt.scatter(pos_x, pos_y, c=range(len(pos_x)))
        plt.xlim(-7000, 7000)
        plt.ylim(-7000, 7000)
        plt.title("Position on the map")
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('Game Steps', rotation=270)
        plt.show()