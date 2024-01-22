import subprocess
import sys


# make a path runs 2-agent_WPM_fully-connected_slow_i where i is the run number
num_test_episodes = 5000
num_agents = 3
num_train_runs = 10
# rsrn_types = ['WSM', 'MinMax']
rsrn_types = ['WPM']
agent_limitations = ['slow'] #'slow', 'stuck'
# networks = ['fully-connected']
networks = ['self-interested',
            'fully-connected',
            'authoritarian',
            'collapsed authoritarian',
            'tribal',
            'collapsed tribal'] #'fully-connected', 'collapsed authoritarian', 'collapsed tribal'


for i in range(num_train_runs):
    loading_paths = []
    for rsrn_type in rsrn_types:
        for network in networks:
            for agent_limitation in agent_limitations:
                loading_paths.append('./saved_policy/' + str(num_agents) + '-agent_' + rsrn_type + '_' + network + '_' + agent_limitation + '_' + str(i+1) + '/')

    # Define the base command and the Python executable
    python_executable = sys.executable
    base_command = [python_executable, "train_v3.py", "--test-mode"]
    commands = [
        base_command + ["--num-episodes", str(num_test_episodes), "--load-dir", path]
        for path in loading_paths
    ]

    # List to keep track of the processes
    processes = []
    # Start all the processes
    for cmd in commands:
        process = subprocess.Popen(cmd)
        processes.append(process)

    for process in processes:
        process.wait()