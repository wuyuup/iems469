# HW2 iems469

HW2 for IEMS-469 (Zuyue Fu)

Run (with CUDA automatically enabled)
```
python /cartpole_reinforce/cartpole.py
```
to obtain the following figure for Cartpole-v0 with REINFORCE: 

![alt text](https://github.com/wuyuup/iems469/blob/master/hw2/cartpole_reinforce/cartpole.png?raw=true)

This is a vanilla implementation of REINFORCE, where the baseline is the mean of the rewards. 

Run (with CUDA disabled but with 16 CPU workers)
```
python /pong_a2c/main.py
```
to obtain the following figure for Pong-v0 with Advantage Actor-Critic: 

![alt text](https://github.com/wuyuup/iems469/blob/master/hw2/pong_a2c/pong.png?raw=true)

This is an implementation of Advantage Actor-Critic, where the baseline is the state value function (which is estimated by the DNN). In terms of the DNN, we use an embedding structure to embed the figure to a feature vector, and we use the feature vector to calculate the probabilities of the actions and the state value function.  Meanwhile, we use 16 workers to train the shared model.  In detail, in the beginning of a training step, the worker loads the parameters of the shared model to its own local model, and then calculate the gradient and update the shared model. This speeds up the convergence (but still not to an optimal policy) with episodic cummulative reward about 10. 

