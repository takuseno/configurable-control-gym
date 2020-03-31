# configurable_control_gym
This repository provides configurable control tasks based on default environments included in [OpenAI Gym](https://github.com/openai/gym).
This work is inspired by [Packer et al, "Assessing Generalization in Deep Reinforcement Learning", 2018](https://arxiv.org/abs/1810.12282).

This program is especially useful to the following deep reinforcement learning research projects:
- assessing generalization performance
- domain randomization
- robust continuous control

## install
```
$ pip install git+https://github.com/takuseno/configurable-control-gym
```

## supported environments
All environments are adapted to continuous action-space even if the original environment supports only discrete action-space.

| name | original environment | configurable parameters (default value) |
|:-:|:-:|:-:|
| ConfigurableCartPole-v0 | CartPole-v0 | force(10.0), length(0.5), mass(0.1) |
| ConfigurablePendulum-v0 | Pendulum-v0 | force(10.0), length(1.0), mass(1.0) |
| ConfigurableAcrobot-v0 | Acrobot-v0 | length(1.0), mass(1.0), moi(1.0) |
| ConfigurableMountainCar-v0 | Continuous_MountainCar-v0 | force(0.001), mass(0.0025) |
| ConfigurableWalker-v0 | BipedalWalker-v2 | torque(80), friction(2.5), length(34) |

## usage
### set fixed parameters
```py
import gym
import configurable_control_gym

env = gym.make('ConfigurableCartPole-v0', length=2.0)
```

### set rages of parametes
Multiple parameter ranges can be set at each parameter.
At every `reset` call, the parameters are uniformly sampled from the given ranges.

```py
import gym
import configurable_control_gym

env = gym.make('ConfigurableCartPole-v0', length=[[0.1, 2.0]],
               mass=[[0.001, 0.1], [0.2, 0.3]])


env.reset() # length is uniformly sampled from [0.1, 2.0]
            #, and mass is uniformly sampled from union of [0.001, 0.1] and [0.2, 0.3].
```
