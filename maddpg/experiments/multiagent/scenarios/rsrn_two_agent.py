import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# np.random.seed(101)
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        world.collaborative = False

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % (i+1)
            agent.collide = True
            agent.silent = True
            agent.size = 0.2
            # if i < 2:
            #     agent.color = np.array([0.9, 0.2, 0.5])
            #     agent.team = 'A'
            # else:
            #     agent.color = np.array([0.35, 0.35, 0.85])
            #     agent.team = 'B'

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
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
        for i, agent in enumerate(world.agents):
            agent.color = np.array(BLUE) ## BLUE
            # if agent.name == 'agent 1':
            #     agent.color = np.array([0.84, 0.15, 0.15]) ## BLUE
            # if agent.name == 'agent 2':

            # if agent.name == 'agent 3':
            #     agent.color = np.array(BLUE) ## BLUE
            # if agent.name == 'agent 4':
            #     agent.color = np.array(BLUE) ## BLUE
            # if agent.name == 'agent 5':
            #     agent.color = np.array(BLUE) ## BLUE
            # if agent.name == 'agent 6':
            #     agent.color = np.array(BLUE) ## BLUE


            if agent.name == 'agent 1':
                agent.color = np.array(BLUE) ## BLUE

            if agent.name == 'agent 2':
                agent.color = np.array(RED) ## BLUE
                agent.max_action_force = 0.1







        # random properties for landmarks
        RED = [0.84, 0.15, 0.15]

        for i, landmark in enumerate(world.landmarks):
            # if i == 0:
            landmark.color = np.array(BLUE)
            # if i == 1:
            #     landmark.color = np.array(RED)
            # if i == 2:
            #     landmark.color = np.array(RED)
            # if i == 3:
            #     landmark.color = np.array(RED)
            # if i == 4:
            #     landmark.color = np.array(RED)
            # if i == 5:
            #     landmark.color = np.array(RED)
            # if i == 6:
            #     landmark.color = np.array(RED)
        #
        # set random initial states

        for i, agent in enumerate(world.agents):


            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)


            # if agent.name == 'agent 1':
            #     agent.state.p_pos = np.array([+0.6, +0.6])
            # if agent.name == 'agent 2':
            #     agent.state.p_pos = np.array([-0.6, +0.6])
            # if agent.name == 'agent 3':
            #     agent.state.p_pos = np.array([-0.6, -0.6])
            # if agent.name == 'agent 4':
            #     agent.state.p_pos = np.array([+0.6, -0.6])
            #




            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # landmark.state.p_vel = np.zeros(world.dim_p)

            # if i == 0:
            #     landmark.state.p_pos = np.array([0.0, -0.6])
            # if i == 1:
            #     landmark.state.p_pos = np.array([+0.6, +0.0])
            # if i == 2:
            #     landmark.state.p_pos = np.array([-0.6, +0.0])
            # if i == 3:
            #     landmark.state.p_pos = np.array([0.0, +0.6])
            r = 0.5
            n = len(world.landmarks)
            # n = 3
            if n == 1:
                landmark.state.p_pos = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = np.array([r*np.cos((i)*(2*np.pi)/n), r*np.sin((i)*(2*np.pi)/n)])


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        # rew = 0
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        test = 0

        rew_type = 'push'

        rew_structure = 'WSM'




        if rew_structure == 'WSM':
            rew = 0
        if rew_structure == 'WPM':
            rew = 1

        if rew_type == 'push' and test == 0:
            if agent.name == 'agent 1':
                network = np.array([1, 1])
            if agent.name == 'agent 2':
                network = np.array([1, 1])


        ### Centeralized Reward
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     # dists = dists * network
        #     rew -= min(dists)
        #     # rew -= np.dot(dists,network)
        #     # rew -= np.dot(dists,network)/np.sum(network)
        # rew = 0


        ### Decenteralized Reward
        # dists = [np.sum(np.square(agent.state.p_pos - l.state.p_pos)) for l in world.landmarks]
        # rew = - min(np.sqrt(dists))

        ### Networked Reward
        if 'network' in locals():
            # count and record the number of pairwise collisions
            # collision_counter = [0] * len(world.agents)
            # if agent.collide:
            #     for i, a in enumerate(world.agents):
            #         if self.is_collision(a, agent):
            #             # red = np.array([0.84, 0.15, 0.15])
            #             # blue = np.array([0.12, 0.46, 0.70])
            #             # rew -= 1
            #             if a.name == 'agent 1':
            #                 collision_counter[0] = 1
            #             if a.name == 'agent 2':
            #                 collision_counter[1] = 1
            #
            # with open(str(rew_type)+'_'+str(agent.name)+"_collision.csv", "a") as f:
            #     f.write(','.join(map(str, collision_counter))+'\n')

            # calculate rewards
            landmark_vector = []
            # collision_vector = np.zeros(len(world.agents))

            for i, a in enumerate(world.agents):
                # dists = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) for l in world.landmarks]
                # landmark_vector.append(-min(np.sqrt(dists)))
                # for j, agent_j in enumerate(world.agents):
                #     if self.is_collision(a, agent_j):
                #         collision_vector[i] = 0

                dists = np.sqrt([np.sum(np.square(a.state.p_pos - l.state.p_pos)) for l in world.landmarks])
                d = min(dists)
                if d < 0.2 or True:
                    landmark_vector.append(max(np.exp(-(d**2)/0.1),0.01))
                    # landmark_vector.append(1)
                else:
                    landmark_vector.append(np.array(0))


            if np.sum(network) != 0:
                # personal_rewards = np.add(np.asarray(landmark_vector),collision_vector)
                personal_rewards = landmark_vector
                # rew = np.dot(personal_rewards,network)/np.sum(network)
                # rew = np.dot(network,personal_rewards)
                # rew = personal_rewards[0] * personal_rewards[1] * personal_rewards[2]


                for k in range(len(world.agents)):
                    if rew_structure == 'WPM':
                        rew = rew * personal_rewards[k]**network[k]
                    if rew_structure == 'WSM':
                        rew = rew + personal_rewards[k]*network[k]
                # rew += np.dot(collision_vector,network)/np.sum(network)



        # rew = rew * np.e**(-episode_step/12)
        # rew = rew * episode_step
        # print(episode_step)
        return rew

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
