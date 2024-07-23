## Week 1:
Q-learning
- Q-learning implementation for gym environment ```Mountain Car Discrete```
- Epsilon Greedy exploration strategy with epsilon decay rate of ```2/episodes``` and floor of epsilon = 0
- Update Q-table values with:
$Q(state,action) = (1 - \alpha)Q(state,action) + \alpha(reward+\gamma argmax_\theta(Q(state,\theta) - Q(state,action)))$
- To handle continuous state environment, organize into buckets for the Q-table to have finite size
- Validation environment runs every 10 runs taking average of 5 attempts with agent taking only deterministic actions
-  Training losses:\
![Training Losses](images/q-learning0.5.png)
- Validation runs tuple list is exported in ```.pkl``` format for use in comparison scripts
## Week 2:
SAC-Discrete
- SAC discrete implentation for same gym environment ```Mountain Car Discrete```
- Utilizes double Q-networks for critic, Agent network, and Target networks
- Update Critic networks with loss function:
$Loss_C = MSE\_loss(Q1, Q\_target1)+MSE\_loss(Q2,Q\_target2)$
- Update Actor network with loss function:
$Q_1, Q_2 = q\_critic(s)$
$Loss\_a = \sum ( \text{probs} \cdot (\alpha \cdot log\_probs - \min (Q_1,Q2)))$

- Update target Q-networks with polyak averaging:
$\theta_{t+1} = \alpha_{t}\theta_{t}\ + (1-\alpha_{t})\theta_{t}$

Comparison with Q-learning
- Utilize matplotlib script to show comparison between two pickled validation episodes of both Q-learning and SAC-Discrete\
![comparison](images/comparison.png)

## Week 3
SAC
- Implemented Continous version of SAC algorithm based on my Discrete implementation
    - Changed output layer of Policy network to output log standard deviation and mean of a normal distribution. Sampling changed accordingly.
    - Critic networks also accept state and action since actions are now a continuous value.
- In the validation, curve due to very different reward structure, it goes from 0 to 100 very quickly as there is a small penalty for taking additional moves but a +100 reward for reaching the end which was not in the Discrete model.\
![SAC](images/SAC_eval.png)

PPO
