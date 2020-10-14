import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

K = 15
# A_t
cf = 100
ch = 2
gamma = 0.95
cmax = 200 # max number of customers
Amin = 1
Amax = 5
max_iter = 1e7
tol = 1e-7
T = 500

# p1

# enumeration
def enumeration(K, cf, ch, max_iter, cmax, tol, Amin, Amax):
    Qfunc = np.array([[0.0 for _ in range(2)] for _ in range(cmax+5)]) # Q(s,a)
    Qfunc_prev = np.array([[0.0 for _ in range(2)] for _ in range(cmax+5)]) # Q(s,a)
    for t in range(T,-1,-1):
        for a in range(2):
            for s in range(cmax+1):
                s_prime = max(0, s - a*K) # next state before incoming customers
                rew = -(cf*a + ch*s)
                # expectation of max
                EQ = 0
                # sample traj
                for A_t in range(Amin, Amax+1):
                    prob = 1.0 / (Amax-Amin+1)
                    new_s = min(s_prime+A_t, cmax)
                    EQ += prob * max(Qfunc_prev[new_s])
                Qfunc[s][a] = rew + gamma * EQ

        cur_error = np.max(abs(Qfunc_prev-Qfunc))
        Qfunc_prev = np.copy(Qfunc)

    return np.max(Qfunc, 1)[:cmax+1]


# VI
def value_iter(K, cf, ch, max_iter, cmax, tol, Amin, Amax):
    Qfunc = np.array([[0.0 for _ in range(2)] for _ in range(cmax+5)]) # Q(s,a)
    Qfunc_prev = np.array([[0.0 for _ in range(2)] for _ in range(cmax+5)]) # Q(s,a)
    cur_error = tol + 10
    niter = 0
    while cur_error > tol and niter < max_iter:
        niter += 1
        for a in range(2):
            for s in range(cmax+1):
                s_prime = max(0, s - a*K) # next state before incoming customers
                rew = -(cf*a + ch*s)
                # expectation of max
                EQ = 0
                # sample traj
                # A_t = sample_A()
                # new_s = min(s_prime+A_t, cmax)
                # EQ += max(Qfunc_prev[new_s])
                for A_t in range(Amin, Amax+1):
                    prob = 1.0 / (Amax-Amin+1)
                    new_s = min(s_prime+A_t, cmax)
                    EQ += prob * max(Qfunc_prev[new_s])
                Qfunc[s][a] = rew + gamma * EQ

        cur_error = np.max(abs(Qfunc_prev-Qfunc))
        Qfunc_prev = np.copy(Qfunc)

    return np.max(Qfunc, 1)[:cmax+1]




# PI

def policy_iter(K, cf, ch, max_iter, cmax, tol, Amin, Amax):

    policy = np.array([-1 for _ in range(cmax+5)])

    Qfunc = np.array([[0.0 for _ in range(2)] for _ in range(cmax+5)]) # Q(s,a)
    Qfunc_prev = np.array([[0.0 for _ in range(2)] for _ in range(cmax+5)]) # Q(s,a)

    cur_error = tol + 10
    niter = 0
    while cur_error > tol and niter < max_iter:
        niter_eval = 0
        cur_error_eval = tol + 10
        niter += 1
        while cur_error_eval > tol and niter_eval < max_iter:
            niter_eval += 1
            for a in range(2):
                for s in range(cmax+1):
                    s_prime = max(0, s - a*K) # next state before incoming customers
                    rew = -(cf*a + ch*s)
                    # expectation of max
                    EQ = 0
                    for incre in range(Amin, Amax+1):
                        prob = 1.0 / (Amax-Amin+1)
                        new_s = min(s_prime+incre, cmax)
                        EQ += prob * Qfunc_prev[new_s][policy[new_s]]
                    Qfunc[s][a] = rew + gamma * EQ
            cur_error_eval = np.max(abs(Qfunc_prev-Qfunc))
            Qfunc_prev = np.copy(Qfunc)

        policy_old = np.copy(policy)
        policy = np.argmax(Qfunc, 1)
        cur_error = np.max(abs(policy - policy_old))
        

    return policy[:cmax+1]


plt.plot(enumeration(K, cf, ch, max_iter, cmax, tol, Amin, Amax))
plt.xlabel('number of customers')
plt.ylabel('value function')
plt.title('value function using enumeration')
plt.savefig('enum.pdf')
plt.clf()

plt.plot(value_iter(K, cf, ch, max_iter, cmax, tol, Amin, Amax))
plt.xlabel('number of customers')
plt.ylabel('value function')
plt.title('value function using VI')
plt.savefig('vi.pdf')
plt.clf()

plt.plot(policy_iter(K, cf, ch, max_iter, cmax, tol, Amin, Amax))
plt.xlabel('number of customers')
plt.ylabel('policy')
plt.title('policy using PI')
plt.savefig('pi.pdf')
plt.clf()

