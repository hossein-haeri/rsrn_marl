import os
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
os.environ['SUPPRESS_MA_PROMPT'] = '1'
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2: WARNING and ERROR messages are not printed
# Suppress specific warnings about Python 3.5 support being deprecated

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import csv
import datetime
import wandb
# np.random.seed(101)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="rsrn_original", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=70, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=500000, help="number of episodes")
    parser.add_argument("--num-agents", type=int, default=3, help="number of agents")
    parser.add_argument("--num-landmarks", type=int, default=3, help="number of landmarks")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # RSRN parameters
    parser.add_argument("--rsrn-type", type=str, default="WPM", help="how RSRN is implemented (WSM, MinMax, WPM)")
    parser.add_argument("--network", type=str, default="fully-connected", help="which network to use (fully-connected, self-interested, authoritarian, ...)")
    parser.add_argument("--agent-limitation", type=str, default="slow", help="is there any limitation on one of the agents?") # normal, slow, stuck 
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    # parser.add_argument("--batch-size", type=int, default=4096, help="number of episodes to optimize at the same time")
    parser.add_argument("--batch-size", type=int, default=2048, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./saved_policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./saved_policy/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(arglist)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))

    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def get_means(ls, chunk_length):
    groups = [ls[x:x+chunk_length] for x in range(0, len(ls), chunk_length)]
    means_list = [sum(group)/len(group) for group in groups]
    return np.asarray(means_list)


def get_STDs(ls, chunk_length):
    groups = [ls[x:x+chunk_length] for x in range(0, len(ls), chunk_length)]
    std_list = [np.std(group) for group in groups]
    return np.asarray(std_list)


def plot_mean_with_std(ls, chunk_length, title, **kwargs):

    scale = 1
    for key, value in kwargs.items():
        if key=='scale':
            scale = value
        if key=='color':
            c = value
    means = get_means(ls, chunk_length)
    stds = get_STDs(ls, chunk_length)
    # plt.plot(means - scale*stds, lw=0.5, c='#396AB1')
    # plt.plot(means + scale*stds, lw=0.5, c='#396AB1')
    plt.fill_between(range(len(means)),means - scale*stds, means + scale*stds, color=c, alpha=0.1)
    plt.plot(means, lw=1, color=c)
    plt.title(str(title))

def plot_rewards(agent_rewards, average_window,title):
    plt.cla()
    colors = [
    '#396AB1',
    '#DA7C30',
    '#3E9651',
    '#CC2529',
    '#535154',
    '#6B4C9A',
    '#922428',
    '#948B3D'
    ]
    for i, reward_list in enumerate(agent_rewards):
        plot_mean_with_std(agent_rewards[i][:], int(average_window), title, scale=0.5, color=colors[i])
        # plt.plot(get_means(agent_rewards[i][:], int(average_window/4)))
    ax = plt.gca()
    ax.set_xlabel('Episodes (x' + str(arglist.save_rate) + ')')
    ax.set_ylabel('Reward')
    plt.grid()
    # plt.legend(['agent 1', 'agent 2', 'agent 3', 'agent 4'])
    plt.legend(['agent 1', 'agent 2', 'agent 3'])
    # plt.legend(['agent 1', 'agent 2'])
    plt.pause(0.0000001)

def train(arglist):
    # csvfile = open('test.csv', 'w', newline='')
    # filewriter = csv.writer(csvfile, delimiter=' ')

    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('initiating {}-agent using {} with {} network'.format(arglist.num_agents, arglist.rsrn_type, arglist.network))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        # episode_rewards = [0.0]  # sum of rewards for all agents
        # agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_rewards = np.zeros((arglist.num_episodes, env.n))
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        # agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        episode_count = 0
        episode_trajectory = []
        trajectories = []
        print('Starting iterations...')
        while True:
            
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n,episode_step)

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                agent_rewards[episode_count, i] = rew


            if done or terminal:
                trajectories.append(episode_trajectory)
                episode_trajectory = []
                obs_n = env.reset()
                episode_step = 0
                episode_count += 1

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(.1)
                env.render()
                continue

            env.world.time = episode_step


            # # update all trainers, if not in display or benchmark mode
            # loss = None
            # for agent in trainers:
            #     agent.preupdate()
            # for agent in trainers:
            #     loss = agent.update(trainers, train_step)

            row = []
            row.append(episode_count+1)
            row.append(episode_step)
            for agent in env.world.agents:
                row.append(agent.state.p_pos[0])
                row.append(agent.state.p_pos[1])
            for landmark in env.world.landmarks:
                row.append(landmark.state.p_pos[0])
                row.append(landmark.state.p_pos[1])
            row.append(rew_n[0])
            row.append(rew_n[1])
            row.append(rew_n[2])
            episode_trajectory.append(row)

            # save model, display training output
            if terminal and (episode_count % arglist.save_rate == 0):
                # U.save_state(arglist.save_dir, saver=saver)

                # mean_rewards = [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards]
                roll_mean_rewards = np.mean(agent_rewards[episode_count-arglist.save_rate:episode_count,:], axis=0)

                # log rewards according to number of agents
                if arglist.num_agents == 2:
                    wandb.log({ "agent 1": roll_mean_rewards[0],
                                "agent 2": roll_mean_rewards[1]})
                elif arglist.num_agents == 3:
                    wandb.log({ "agent 1": roll_mean_rewards[0],
                                "agent 2": roll_mean_rewards[1],
                                "agent 3": roll_mean_rewards[2]})

                # pickle.dump(agent_rewards, open(str(arglist.save_dir)+'rewards.pkl', 'wb'))
                with open("./test_policy/" + arglist.exp_name + '/test_trajectory.pkl', 'wb') as file:
                    pickle.dump(trajectories, file)
                
                print("episodes: {}, agent episode reward: {}, time: {}".format(
                    episode_count,
                    roll_mean_rewards,
                    round(time.time()-t_start, 3)))
                t_start = time.time()

            if episode_count >= arglist.num_episodes:
                print('...Finished total of {} episodes.'.format(episode_count))
                break

if __name__ == '__main__':
    arglist = parse_args()
    arglist.save_dir = "./test_policy/"+str(arglist.num_agents)+"-agent/"+arglist.rsrn_type+"/"+arglist.network+"/"+str(arglist.exp_name)+"/"
    # save all arguments to csv file in a new directory (arglist.save_dir)
    if not os.path.exists(arglist.save_dir):
        os.makedirs(arglist.save_dir)
    with open(arglist.save_dir+'hyperparams.txt', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=':')
        for key, value in vars(arglist).items():
            filewriter.writerow([key, value])
    # print(arglist)


    wandb.init(project='RSRN', entity='haeri-hsn')
    config = wandb.config
    config.network = arglist.network
    config.num_agents = arglist.num_agents
    config.num_landmarks = arglist.num_landmarks
    config.agent_limitation = arglist.agent_limitation
    config.rsrn_type = arglist.rsrn_type
    config.boundary = '(-1.2,1.2)'
    config.learning_rate = arglist.lr
    config.gamma = arglist.gamma
    config.batch_size = arglist.batch_size
    config.num_units = arglist.num_units
    config.num_episodes = arglist.num_episodes
    config.max_episode_len = arglist.max_episode_len
    config.good_policy = arglist.good_policy
    config.adv_policy = arglist.adv_policy
    config.exp_name = arglist.exp_name
    config.save_dir = arglist.save_dir
    config.save_rate = arglist.save_rate



    train(arglist)
    wandb.finish()

    plt.show()
