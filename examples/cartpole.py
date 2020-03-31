import gym
import configurable_control_gym


env = gym.make('ConfigurableCartPole-v0', length=2.0)

env.reset()
while True:
    _, _, done, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        break
