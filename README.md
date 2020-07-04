[![<abstractpaper>](https://circleci.com/gh/abstractpaper/prop.svg?style=shield)](https://circleci.com/gh/abstractpaper/prop)

# prop

prop is a library of Reinforcment Learning agents implemented in pytorch.

## Algorithms

|         | Model      | Policy     |
|---------|------------|------------|
| **DQN** | Model-Free | Off-Policy |
| **A2C** | Model-Free | On-Policy  |

## DQN

[Deep Q-Learning][1] is a variant of Q-learning with a deep neural network used for estimating Q-values (hence DQN; Deep Q-Network).

Both DQN and DDQN (Double DQN) are implemented.

## A2C

[Advantage Actor Critic][2] is a variant of Actor-Critic that:
- Uses a neural network to approximate a policy and a value function.
- Computes the advantage of an action to scale the computed gradients. This acts as a vote of confidence (or skepticism) on actions produced by the actor.

[1]: https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning
[2]: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752
[3]: https://github.com/astooke/rlpyt
[4]: https://github.com/openai/baselines
[5]: https://github.com/NervanaSystems/coach
