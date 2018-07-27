import math
import numpy as np

class UCB1():
    def __init__(self, n_arms, alpha=2):
        self.counts = np.zeros(n_arms)
        self.sum = np.zeros(n_arms)
        self.n_arms = n_arms
        self.ucb_values = np.zeros(n_arms)
        self.alpha = alpha

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        return np.argmax(self.ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.sum[chosen_arm] += reward

        #calculate new UCB1 value for chosen_arm
        total_counts = np.sum(self.counts)
        mean_for_chosen_arm = self.sum[chosen_arm] / float(self.counts[chosen_arm])
        self.ucb_values[chosen_arm] = mean_for_chosen_arm + self.alpha * math.sqrt((2 * math.log(total_counts)) / float(self.counts[chosen_arm]))