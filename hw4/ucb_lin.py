import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import tensorflow.compat.v1 as tf
import random
import logging
import argparse

tf.disable_v2_behavior()

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--log-dir', default = 'regret.log', help='directory to save log file')
parser.add_argument('--alpha', default = 200.0, type=float)
parser.add_argument('--nepisode', default = 200, type=int)


logging.basicConfig(filename = parser.parse_args().log_dir, filemode='w', level=logging.INFO)



def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181,
    shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.
    Args:
       file_name: Route of file containing the modified Jester dataset.
       context_dim: Context dimension (i.e. vector with some ratings from a user).
       num_actions: Number of actions (number of joke ratings to predict).
       num_contexts: Number of contexts to sample.
       shuffle_rows: If True, rows from original dataset are shuffled.
       shuffle_cols: Whether or not context/action jokes are randomly shuffled.
    Returns:
       dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
       opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """
    np.random.seed(0)
    with tf.gfile.Open(file_name, 'rb') as f:
        dataset = np.load(f)
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])
    return dataset, opt_rewards, opt_actions



d = 32 # dimension of feature
alpha = parser.parse_args().alpha
lamd = 1.0
A = np.zeros((8, d, d))
for a in range(8):
    A[a] = np.eye(d) * lamd

b = np.zeros((8, d))
p = np.zeros(8)
beta = 0.000001
data = sample_jester_data('jester_data_40jokes_19181users.npy')


def feature_embed(x, a):
    return x

def ucb_step(ind):
    # sample data x
    # ind = np.random.randint()
    # context
    x = data[0][ind][:32]
    for a in range(8):
        # embedding
        phi = feature_embed(x,a)
        # matrix
        inv_A = np.linalg.inv(A[a])
        theta = np.dot(inv_A, b[a])
        # bonus
        bonus = np.sqrt(np.dot(np.dot(phi, inv_A), phi))
        # mean 
        mu_mean = np.dot(theta, phi)
        p[a] = mu_mean + alpha * bonus + np.random.rand() * beta   #  break tie

    # choose action
    choice_a = p.argmax()
    # action + 8 to get reward
    reward = data[0][ind][int(choice_a)+32]
    # optimal 
    optimal_reward = data[1][ind]
    # regret
    cur_regret = optimal_reward - reward

    # update
    A[int(choice_a)] += np.outer(feature_embed(x,choice_a), feature_embed(x,choice_a))
    b[int(choice_a)] += reward * feature_embed(x,choice_a)

    return cur_regret



total_regret = 0
# f = open('regret.txt', 'w')

# for ind in range(18000):
for nstep in range(18000 * parser.parse_args().nepisode):
    ind = np.random.randint(18000)
    cur_regret = ucb_step(ind)
    total_regret += cur_regret
    output_str = str(total_regret) + ' ' + str(total_regret/nstep)
    if nstep % 1000 == 0:
        print(output_str)
    # logging.info(output_str)
    # f.write(str(total_regret) + '\n')
    # print(total_regret)


total_regret = 0

for ind in range(18000, 19181):
    cur_regret = ucb_step(ind)
    total_regret += cur_regret
    output_str = str(total_regret) + ' ' + str(total_regret/nstep)
    logging.info(output_str)

logging.close()





