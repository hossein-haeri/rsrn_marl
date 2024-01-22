import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# np.random.seed(101)
class Scenario(BaseScenario):
    def make_world(self, arglist):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.collaborative = False
        world.num_agents = arglist.num_agents
        world.num_landmarks = arglist.num_landmarks
        world.agent_limitation = arglist.agent_limitation
        world.network = arglist.network
        world.rsrn_type = arglist.rsrn_type
        world.stuck_location = arglist.stuck_location
        # add agents
        world.agents = [Agent() for i in range(world.num_landmarks)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % (i+1)
            agent.id = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.2
            agent.indiviual_reward = None

        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_agents)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        BLUE = [0, 0.4470, 0.7410]
        RED = [0.8500, 0.3250, 0.0980]
        YELLOW = [0.929, 0.6940, 0.1250]
        GRAY = [0.5, 0.5, 0.5]

        for i, agent in enumerate(world.agents):
            agent.color = np.array(GRAY) 
            if agent.name == 'agent 1':
                agent.color = np.array(BLUE) ## BLUE
            if agent.name == 'agent 2':
                agent.color = np.array(RED) ## BLUE
            if agent.name == 'agent 3':
                agent.color = np.array(YELLOW) ## BLUE
        
        # random properties for landmarks
        RED = [0.84, 0.15, 0.15]

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array(BLUE)

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            r = 0.5
            n = len(world.landmarks)
            if n == 1:
                landmark.state.p_pos = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = np.array([r*np.cos((i)*(2*np.pi)/n), r*np.sin((i)*(2*np.pi)/n)])
        
        if world.agent_limitation == 'slow':
            world.agents[-1].max_action_force = 0.01

        if world.agent_limitation == 'stuck':
            world.agents[-1].max_speed = 0.0
            world.agents[-1].state.p_pos = world.stuck_location


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    def reward(self, agent, world):

        if world.network == 'self-interested':
            if agent.name == 'agent 1':
                network = np.array([1, 0, 0])
            if agent.name == 'agent 2':
                network = np.array([0, 1, 0])
            if agent.name == 'agent 3':
                network = np.array([0, 0, 1])

        if world.network == 'fully-connected':
            if agent.name == 'agent 1':
                network = np.array([1, 1, 1])
            if agent.name == 'agent 2':
                network = np.array([1, 1, 1])
            if agent.name == 'agent 3':
                network = np.array([1, 1, 1])

        if world.network == 'authoritarian':
            if agent.name == 'agent 1':
                network = np.array([1, 0, 1])
            if agent.name == 'agent 2':
                network = np.array([0, 1, 1])
            if agent.name == 'agent 3':
                network = np.array([0, 0, 1])

        if world.network == 'tribal':
            if agent.name == 'agent 1':
                network = np.array([1, 1, 0])
            if agent.name == 'agent 2':
                network = np.array([0, 1, 1])
            if agent.name == 'agent 3':
                network = np.array([1, 0, 1])
        
        if world.network == 'collapsed authoritarian':
            if agent.name == 'agent 1':
                network = np.array([0, 0, 1])
            if agent.name == 'agent 2':
                network = np.array([0, 0, 1])
            if agent.name == 'agent 3':
                network = np.array([0, 0, 1])

        if world.network == 'collapsed tribal':
            if agent.name == 'agent 1':
                network = np.array([0, 1, 0])
            if agent.name == 'agent 2':
                network = np.array([0, 0, 1])
            if agent.name == 'agent 3':
                network = np.array([1, 0, 0])

        assert sum(network) != 0
        # evaluate individual rewards according to the distance to the closest landmarks
        base_reward = 0.01
        individual_rewards = []
        for i, a in enumerate(world.agents):
            dists = np.sqrt([np.sum(np.square(a.state.p_pos - l.state.p_pos)) for l in world.landmarks])
            d = min(dists)
            if d < 0.2 or True:
                individual_rewards.append(max(np.exp(-(d**2)/0.1), base_reward))
            else:
                individual_rewards.append(np.array(0))
            if i == agent.id:
                agent.individual_reward = individual_rewards[agent.id]
                agent.dist2landmark = d



        if world.rsrn_type == 'WSM':
            shared_reward = 0
            for k in range(len(world.agents)):
                shared_reward = shared_reward + individual_rewards[k]*network[k]
        if world.rsrn_type == 'WPM':
            shared_reward = 1
            for k in range(len(world.agents)):
                shared_reward = shared_reward * individual_rewards[k]**network[k]
        if world.rsrn_type == 'MinMax':
            priorities = []
            for k in range(len(world.agents)):
                if network[k] == 1:
                    priorities.append(individual_rewards[k])
            shared_reward = min(priorities)
 
 
        return shared_reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []

        for i, other in enumerate(world.agents):
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # np.random.shuffle(entity_pos)
        # for entity in world.agents:
        # print('start')
        # print(world.time)
        # print('end')
            # print(entity.name)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + np.asarray(world.time) + np.asarray(world.disabled_agent_num))
