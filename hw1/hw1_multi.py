import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import copy


K = 30
# A_t
cf = 100
ch = [1.0, 1.5, 2.0, 2.5, 3.0]
gamma = 0.95
cmax = 100 # max number of customers
Amin = 1
Amax = 5
max_iter = 1e7
tol = 1e-7
T = 500



# util

# construct state space
state_space = []
def gen_state_space(cur):
    if len(cur) == 5:
        state_space.append(tuple(cur))
    else:
        for i in range(cmax+1):
            gen_state_space(cur+[i])

gen_state_space([])


# generate scenarios
scenario = []
def gen_scenario(cur):
    if len(cur) == 5:
        scenario.append(tuple(cur))
    else:
        for i in range(Amin, Amax+1):
            gen_scenario(cur+[i])

gen_scenario([])


def def_value(): 
    return [0.0, 0.0]

def def_value_v(): 
    return 0.0

def next_state_before_arrival(s, a):
    news = np.copy(s)
    if a == 1: # dispatch
        # first dispatch large cost
        left = K
        for i in range(len(news)-1,-1,-1):
            if left > 0:
                avail = min(news[i], left)
                news[i] -= avail
                left -= avail
            else:
                break
        # possibly left > 0 which indicates nobody in the queue
    return news


def next_state_after_arrival(s, A): # A: vector of arrivals
    news = s+A
    for i,num in enumerate(news):
        if num > cmax:
            news[i] = cmax
    return news


def rew_func(s, a):
    res = -cf*a
    for i,item in enumerate(s):
        res -= item * ch[i]
    return res






# p2

# enumeration
def enumeration(K, cf, ch, max_iter, cmax, tol, Amin, Amax):  # return a dict
    Qfunc = defaultdict(def_value)  # Qfunc[s] is a vector of length 2
    Qfunc_prev = defaultdict(def_value)
    Vfunc = defaultdict(def_value_v)

    for t in range(T,-1,-1):
        for a in range(2):
            for s in state_space:
                s_prime = next_state_before_arrival(s, a) # next state before incoming customers
                rew = rew_func(s, a)
                # expectation of max
                EQ = 0
                for A_t in scenario: # check each scenario
                    prob = 1.0 / len(scenario)
                    new_s = next_state_after_arrival(s_prime, A_t)
                    EQ += prob * max(Qfunc_prev[new_s])
                Qfunc[s][a] = rew + gamma * EQ
                Vfunc[s] = max(Qfunc[s])
        Qfunc_prev = copy(Qfunc)
    return Vfunc


# VI
def value_iter(K, cf, ch, max_iter, cmax, tol, Amin, Amax):
    Qfunc = defaultdict(def_value)  # Qfunc[s] is a vector of length 2
    Qfunc_prev = defaultdict(def_value)
    Vfunc = defaultdict(def_value_v)

    cur_error = tol + 10
    niter = 0
    while cur_error > tol and niter < max_iter:
        niter += 1
        cur_error = 0
        for a in range(2):
            for s in state_space:
                s_prime = next_state_before_arrival(s, a) # next state before incoming customers
                rew = rew_func(s, a)
                # expectation of max
                EQ = 0
                for A_t in scenario: # check each scenario
                    prob = 1.0 / len(scenario)
                    new_s = next_state_after_arrival(s_prime, A_t)
                    EQ += prob * max(Qfunc_prev[new_s])
                Qfunc[s][a] = rew + gamma * EQ
                Vfunc[s] = max(Qfunc[s])
                cur_error = max(cur_error, abs(Qfunc_prev[s][a]-Qfunc[s][a]))
        Qfunc_prev = copy(Qfunc)

    return Vfunc






# PI

def policy_iter(K, cf, ch, max_iter, cmax, tol, Amin, Amax):

    policy = defaultdict(def_value_v)
    Qfunc = defaultdict(def_value)  # Qfunc[s] is a vector of length 2
    Qfunc_prev = defaultdict(def_value)

    cur_error = tol + 10
    niter = 0
    while cur_error > tol and niter < max_iter:
        niter += 1
        cur_error = 0
        for a in range(2):
            for s in state_space:
                s_prime = next_state_before_arrival(s, a) # next state before incoming customers
                rew = rew_func(s, a)
                # expectation of max
                EQ = 0
                for A_t in scenario: # check each scenario
                    prob = 1.0 / len(scenario)
                    new_s = next_state_after_arrival(s_prime, A_t)
                    EQ += prob * Qfunc_prev[new_s][policy[new_s]]
                Qfunc[s][a] = rew + gamma * EQ
                cur_error = max(cur_error, abs(Qfunc_prev[s][a]-Qfunc[s][a]))
        for s in state_space:
            policy = np.argmax(Qfunc[s])
        Qfunc_prev = copy(Qfunc)

    return policy




plt.plot(enumeration(K, cf, ch, max_iter, cmax, tol, Amin, Amax))
plt.savefig('pi.pdf')
plt.clf()

plt.plot(value_iter(K, cf, ch, max_iter, cmax, tol, Amin, Amax))
plt.savefig('pi.pdf')
plt.clf()

plt.plot(policy_iter(K, cf, ch, max_iter, cmax, tol, Amin, Amax))
plt.savefig('pi.pdf')
plt.clf()

