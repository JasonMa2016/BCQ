import numpy as np
import torch
import gym


def evaluate_model(env, model, running_state=None, num_trajs=50, verbose=True, render=False):
    """seeding"""
    np.random.seed(2020)
    torch.manual_seed(2020)
    env.seed(2020)
    episodes_rewards = []
    episodes_timesteps = []
    for i in range(num_trajs):
        state = env.reset()
        state = running_state(state)

        episode_rewards = 0
        episode_steps = 0
        for t in range(10000):
            state_var = torch.tensor(state).unsqueeze(0)
            with torch.no_grad():
                # select deterministically
                action = model(state_var)[0][0].numpy()
                # action = imitator.select_action(state_var)[0].numpy()
            action = int(action) if model.is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            episode_rewards += reward
            episode_steps += 1
            if render:
                env.render()
            if done:
                break
            state = next_state

        episodes_rewards.append(episode_rewards)
        episodes_timesteps.append(episode_steps)
        # if args.verbose:
        #     print('{}\tsteps: {}\t reward: {:.2f}'.format(
        #         i, episode_steps, episode_rewards))

    episodes_rewards = np.array(episodes_rewards)
    episodes_timesteps = np.array(episodes_timesteps)
    if verbose:
        print("{} Trajectories \t reward avg: {:.2f} \t reward std: {:.2f}".format(num_trajs,
                                                                       episodes_rewards.mean(),
                                                                       episodes_rewards.std()))
    return {'episodes_rewards': episodes_rewards,
            'episodes_timesteps': episodes_timesteps}


def evaluate_model_atari(env, model, num_trajs=5, verbose=False, render=False):
    episode_rewards = []
    episode_timesteps = []
    for i_episode in range(num_trajs):
        obs = env.reset()
        while True:
            with torch.no_grad():
                action = model(torch.DoubleTensor(obs))

            # select deterministic action
            action = np.argmax(action.detach().numpy())

            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, done, infos = env.step([action])
            if render:
                env.render('human')

            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            episode_infos = infos[0].get('episode')
            if episode_infos is not None and done:
                episode_reward = episode_infos['r']
                episode_length = episode_infos['l']

                episode_rewards.append(episode_reward)
                episode_timesteps.append(episode_length)
                if verbose:
                    print("Episode {}".format(i_episode + 1))
                    print("Atari Episode Score: {:.2f}".format(episode_reward))
                    print("Atari Episode Length", episode_length)
                break
    if verbose:
        print("---------------------------------------")
        print("Mean episode reward: {:.2f}".format(np.mean(episode_rewards)))
        print("Mean episode length: {:.2f}".format(np.mean(episode_timesteps)))
    env.close()
    episode_rewards = np.array(episode_rewards)
    episode_timesteps = np.array(episode_timesteps)

    return {'episodes_rewards': episode_rewards,
            'episodes_timesteps': episode_timesteps}