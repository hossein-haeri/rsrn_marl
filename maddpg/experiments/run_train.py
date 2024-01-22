import subprocess
import sys


# make a path runs 2-agent_WPM_fully-connected_slow_i where i is the run number
num_train_episodes = 500000
save_rate = 1000
num_agents = 3
num_train_runs = 7
rsrn_types = ['WSM', 'MinMax', 'WPM'] # 'WSM', 'MinMax', 'WPM'
agent_limitations = ['normal', 'stuck', 'slow'] #, ['normal', 'stuck', 'slow']

networks = ['fully-connected']
# networks = ['self-interested',
#             'fully-connected',
#             'authoritarian',
#             'collapsed authoritarian',
#             'tribal',
#             'collapsed tribal'] #'fully-connected', 'collapsed authoritarian', 'collapsed tribal'


num_prev_runs = 3

for i in range(num_train_runs):
    commands = []
    for rsrn_type in rsrn_types:
        for network in networks:
            for agent_limitation in agent_limitations:
                if agent_limitation == 'slow' and rsrn_type == 'WPM':
                    continue
                saving_path = './saved_policy/' + str(num_agents) + '-agent_' + rsrn_type + '_' + network + '_' + agent_limitation + '_' + str(num_prev_runs+i+1) + '/'

                commands.append([   "python", "train_v3.py", 
                                    "--num-episodes", str(num_train_episodes), 
                                    "--save-dir", saving_path,
                                    "--save-rate", str(save_rate),
                                    "--num-agents", str(num_agents),
                                    "--num-landmarks", str(num_agents),
                                    "--rsrn-type", rsrn_type,
                                    "--network", network,
                                    "--agent-limitation", agent_limitation,
                                    "--exp-name", agent_limitation + '_' + str(num_prev_runs+i+1)])


                
# python train_v3.py --num-agents 2 --num-landmarks 2 --rsrn-type WPM --network fully-connected --num-episodes $num_episodes --exp-name slow_$i  &


    python_executable = sys.executable

    # List to keep track of the processes
    processes = []
    # Start all the processes
    for cmd in commands:
        process = subprocess.Popen(cmd)
        processes.append(process)

    for process in processes:
        process.wait()