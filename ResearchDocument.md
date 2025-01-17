# Research Document

## Please add your research here:

## Reinforcement Learning

### High-Level Overview
Machine learning algorithm where an **agent** (neural network) learns to make decisions by performing actions in given states and receiving rewards or penalties based on its actions. Essentially, it's a ‘brute force’ approach — the agent learns by trying different actions and understanding the rewards associated with them.

### Key Concepts
- **Agent**: The decision-maker (neural network).
- **Environment**: The system on which the agent performs actions.
- **Reward**: Feedback the agent receives based on the outcome of its actions.
- **Discount Factor**: A parameter balancing immediate and future rewards, allowing the agent to weigh long-term consequences but giving more emphasis to immediate rewards as needed.

### RL Algorithms

#### Policy Learning
- The neural network directly learns the policy from the data.
- Solves a stochastic optimization problem.
- Works well with continuous action spaces.

#### Value Learning (Q-Learning)
- Learns the **Q function**.
- The Q function takes in the current state and a possible action (which can be executed in the current state) and outputs the expected reward for that action.
- Chooses actions that maximize future rewards.
- Only applicable to discrete action spaces.

### Exploration Strategies
To determine when the agent should ‘explore’ (try new actions for a given state) and when to ‘exploit’ (choose actions it knows to yield high rewards in a given state):

- **ϵ-greedy**: A common strategy where, with probability \(1 - ϵ\), the agent will exploit (choose the action with the highest estimated future reward) and with probability \(ϵ\), it will explore (take a random action).

### When to Use RL
- **Sequential Decision-Making**: When actions affect future states, e.g., games, robotics, autonomous vehicles.
- **Changing Environments**: When actions cause dynamic changes in the environment.
- **Absence of Labeled Data**: RL does not rely on labeled data.
- **Cost of Rewards**: When obtaining rewards for specific actions is not prohibitively expensive.
- **Efficiency**: Note that RL is generally less efficient than supervised learning, as it involves more trial and error.

### Challenges with RL
- **Sample Inefficiency**: RL often requires large amounts of data to learn effectively.
- **Generalization**: Difficulty in transferring learned behaviors to new, unseen environments.
- **Credit Assignment Problem**: Determining which specific actions contributed to future rewards and which did not.
- **Exploration / Exploitation Tradeoff**: Balancing trying new actions with utilizing known successful actions.

---

### Sources
- [Good explanation on what RL is with an example](http://karpathy.github.io/2016/05/31/rl/)
- [Overview of RL lecture](https://www.youtube.com/watch?v=AhyznRSDjw8&t=2978s)
- [Application of RL in cybersecurity attacks](https://sciencesforce.com/index.php/aics/article/view/298/459)
- [Exploration / Exploitation methods](https://arxiv.org/pdf/2109.00157)
