from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from q2_REINFORCE import *
matplotlib.use('Agg')
np.random.seed(1234)

'''
For REINFORNCE with baselines Monte-Carlo Policy-Gradient Control (episodic) in LN17.
'''


class ReinforceBaseline(Reinforce):
    """
    Reinforce with baselines that follows algorith 'REINFORNCE with with baselines Monte-Carlo Policy-Gradient Control (episodic)' in LN17.
    Note that in this problem, all the states appear identical under the function approximation.
    To simplify your implementation, here we assume that V(s,w)=w where w is a scalar.
    Thus, h(s,a)=h(a), x(s,a)=x(a), pi(s,a)=pi(a), V(s,w)=w.
    """

    def __init__(self, alpha, gamma, alpha_w):
        super(ReinforceBaseline, self).__init__(alpha, gamma)
        self.alpha_w = alpha_w  # learning rate for w
        self.w = 0

    def learn(self):
        # Update policy

        # Compute discounted return
        G = self.compute_discounted_return()

        # Update theta and w
        for i in range(len(G)):
            ############################
            # YOUR IMPLEMENTATION HERE #
            # TODO Update w
            self.w += self.alpha_w*(G[i]-self.w)
            ############################

            ############################
            # YOUR IMPLEMENTATION HERE #
            # TODO Update theta
            gradient = self.x[self.actions[i]]-np.exp(self.theta)/sum(np.exp(self.theta))
            self.theta += self.alpha*self.gamma**i*(G[i]-self.w)*gradient
            ############################

        self.rewards = []
        self.actions = []


def REINFORCE_with_baselines_plot():
    num_trials = 100
    num_episodes = 1000
    alpha = 2e-4
    gamma = 1
    agent_generators = [lambda: Reinforce(alpha=alpha, gamma=gamma),
                        lambda: ReinforceBaseline(alpha=alpha*10, gamma=gamma, alpha_w=alpha*100)]
    labels = ['Reinforce without baseline',
              'Reinforce with baseline']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))
    policies = np.zeros((len(agent_generators), num_trials, 2))
    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward, policy = run(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward
            policies[agent_index, i, :] = policy

    plt.plot(np.arange(num_episodes) + 1, -11.6 *
             np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    template = 'Final Optimal Policy for {}: Left {}, Right {}'
    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1,
                 rewards[i].mean(axis=0), label=label)
        print(template.format(label, policies[i].mean(
            axis=0)[0], policies[i].mean(axis=0)[1]))
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('img/Q3_reinforce_with_baselines.png')
    plt.close()

    n_lable = len(labels)
    left_p = [policies[i, :, 0].mean(axis=0) for i in range(n_lable)]
    right_p = [policies[i, :, 1].mean(axis=0) for i in range(n_lable)]
    plt.bar(range(n_lable), left_p, label='Left', fc='y')
    plt.bar(range(n_lable), right_p, bottom=left_p,
            label='Right', tick_label=labels, fc='b')
    plt.legend()
    plt.savefig('img/Q3_reinforce_with_baselines_policy.png')
    plt.close()

# Test and run your code. Plot is stored into img/Q3_reinforce_with_baselines.png and img/Q3_reinforce_with_baselines_policy.png.
# You may change the parameters in the functions below.
if __name__ == '__main__':
    REINFORCE_with_baselines_plot()
