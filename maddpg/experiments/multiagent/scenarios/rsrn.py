import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            if i == 0:
                # make color blue for agent 0
                agent.color = np.array([0.35, 0.35, 0.85])
            elif i == 1:
                # make color red for agent 1
                agent.color = np.array([0.85, 0.35, 0.35])
            # limit speed for agent 2
            elif i == 2:
                agent.max_action_force = 0.1
                # make color yelow for agent 2
                agent.color = np.array([0.85, 0.85, 0.35])
            else:
                raise ValueError("agent number not defined!")
    
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        # for i, agent in enumerate(world.agents):
        #     agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # set landmark positions at fixed locations at (-0.5, 0) and (+0.5, 0) and (0, +0.5)
            if i == 0:
                landmark.state.p_pos = np.array([-0.5, 0])
            elif i == 1:
                landmark.state.p_pos = np.array([+0.5, 0])
            elif i == 2:
                landmark.state.p_pos = np.array([0, +0.5])



            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     min_dists += min(dists)
        #     rew -= min(dists)
        #     if min(dists) < 0.1:
        #         occupied_landmarks += 1
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        #             collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)

        # make a 3 by 3 adjacency matrix as reward sharing relational networks (RSRN)
        # A = np.zeros((3,3))
        
        # # self-interested RSRN
        # A =np.array([[1, 0, 0],
        #              [0, 1, 0],
        #              [0, 0, 1]])

        # fully cooperative RSRN
        A = np.array(  [[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])

        # # authoritarian RSRN (agent 2 is the leader)
        # A = np.array([[1, 0, 1],
        #               [0, 1, 1],
        #               [0, 0, 1]])

        # # collapsed authoritarian RSRN (agent 2 is the leader)
        # A = np.array([  [0, 0, 1],
        #                 [0, 1, 0],
        #                 [0, 0, 1]])

        if agent.id == 0:
            for agnt in world.agents:
                # calculate individual reward for each agent
                # find the closest landmark to agent
                # dist = min([np.sqrt(np.sum(np.square(l.state.p_pos - agnt.state.p_pos))) for l in world.landmarks])
                dist = min([np.linalg.norm(l.state.p_pos - agnt.state.p_pos) for l in world.landmarks])
                # reward agent for reaching any landmark with d < 0.1 with reward exp(-dist^2)/sigma^2
                agnt.indiviual_reward = 0.1
                if dist < 0.2:
                        agnt.indiviual_reward = agnt.indiviual_reward + np.exp(-dist**2/0.1)
                # else:
                #     agnt.indiviual_reward = 0

        

        # construct a shared reward as rew_shared as a function of weighted product of the individual rewards already calculated (rew)
        rew_shared = 1
        for a in world.agents:
            rew_shared = rew_shared * (a.indiviual_reward ** A[agent.id, a.id])





        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= .1
        return rew_shared

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
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos)
