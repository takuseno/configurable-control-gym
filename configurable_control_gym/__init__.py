from gym.envs.registration import register

register(
    id='ConfigurablePendulum-v0',
    entry_point='configurable_control_gym.envs.pendulum:PendulumEnv',
    max_episode_steps=200
)

register(
    id='ConfigurableCartPole-v0',
    entry_point='configurable_control_gym.envs.cartpole:CartPoleEnv',
    max_episode_steps=200
)

register(
    id='ConfigurableAcrobot-v0',
    entry_point='configurable_control_gym.envs.acrobot:AcrobotEnv',
    max_episode_steps=500
)

register(
    id='ConfigurableMountainCar-v0',
    entry_point='configurable_control_gym.envs.mountain_car:Continuous_MountainCarEnv',
    max_episode_steps=999
)

register(
    id='ConfigurableBipedalWalker-v0',
    entry_point='configurable_control_gym.envs.walker:BipedalWalker',
    max_episode_steps=1600
)
