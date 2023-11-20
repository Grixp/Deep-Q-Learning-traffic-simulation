import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action, action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads_11 = ['0111', '2111', '1011', '1211']
        incoming_roads_21 = ['1121', '3121', '2021', '2221']
        incoming_roads_12 = ['0212', '2212', '1112', '1312']
        incoming_roads_22 = ['1222', '3222', '2122', '2322']
        incoming_roads = incoming_roads_11+incoming_roads_21+incoming_roads_12+incoming_roads_22
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    def _set_yellow_phase(self, old_action, new_action):
        """
        Activate the correct yellow light combination in sumo
        """
        old_action_11 = int(old_action/64)
        old_action_12 = int((old_action-old_action_11*64)/16)
        old_action_21 = int((old_action-old_action_11*64-old_action_12*16)/4)
        old_action_22 = int(old_action-old_action_11*64-old_action_12*16-old_action_21*4)

        new_action_11 = int(new_action/64)
        new_action_12 = int((new_action-new_action_11*64)/16)
        new_action_21 = int((new_action-new_action_11*64-new_action_12*16)/4)
        new_action_22 = int(new_action-new_action_11*64-new_action_12*16-new_action_21*4)

        if old_action_11 != new_action_11:
            yellow_phase_code_11 = old_action_11 * 2 + 1
            traci.trafficlight.setPhase('11', yellow_phase_code_11)
        if old_action_12 != new_action_12:
            yellow_phase_code_12 = old_action_12 * 2 + 1
            traci.trafficlight.setPhase('12', yellow_phase_code_12)
        if old_action_21 != new_action_21:
            yellow_phase_code_21 = old_action_21 * 2 + 1
            traci.trafficlight.setPhase('21', yellow_phase_code_21)
        if old_action_22 != new_action_22:
            yellow_phase_code_22 = old_action_22 * 2 + 1
            traci.trafficlight.setPhase('22', yellow_phase_code_22)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        
        action_11 = int(action_number/64)
        action_12 = int((action_number-action_11*64)/16)
        action_21 = int((action_number-action_11*64-action_12*16)/4)
        action_22 = int(action_number-action_11*64-action_12*16-action_21*4)

        traci.trafficlight.setPhase('11',2*action_11)
        traci.trafficlight.setPhase('12',2*action_12)
        traci.trafficlight.setPhase('21',2*action_21)
        traci.trafficlight.setPhase('22',2*action_22)
        

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        edges=['0111','1101','1121','2111','2131','3121','1011','1101','2021','2120','0212','1202',\
            '1222','2212','2232','3222','1112','1211','2122','2221','1213','1312','2223','2322']
        queue_length = 0
        for edge in edges:
            queue_length+=traci.edge.getLastStepHaltingNumber(edge)
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = abs(100 - lane_pos)  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 100 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 10:
                lane_cell = 0
            elif lane_pos < 20:
                lane_cell = 1
            elif lane_pos < 30:
                lane_cell = 2
            elif lane_pos < 40:
                lane_cell = 3
            elif lane_pos < 50:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 70:
                lane_cell = 6
            elif lane_pos < 80:
                lane_cell = 7
            elif lane_pos < 90:
                lane_cell = 8
            elif lane_pos < 100:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "0111_0":
                lane_group = 0
            elif lane_id == "0111_1":
                lane_group = 1
            elif lane_id == "2111_0":
                lane_group = 2
            elif lane_id == "2111_1":
                lane_group = 3
            elif lane_id == "1011_0":
                lane_group = 4
            elif lane_id == "1011_1":
                lane_group = 5
            elif lane_id == "1211_0":
                lane_group = 6
            elif lane_id == "1211_1":
                lane_group = 7
            elif lane_id == "1121_0":
                lane_group = 8
            elif lane_id == "1121_1":
                lane_group = 9
            elif lane_id == "3121_0":
                lane_group = 10
            elif lane_id == "3121_1":
                lane_group = 11
            elif lane_id == "2021_0":
                lane_group = 12
            elif lane_id == "2021_1":
                lane_group = 13
            elif lane_id == "2221_0":
                lane_group = 14
            elif lane_id == "2221_1":
                lane_group = 15
            elif lane_id == "0212_0":
                lane_group = 16
            elif lane_id == "0212_1":
                lane_group = 17
            elif lane_id == "2212_0":
                lane_group = 18
            elif lane_id == "2212_1":
                lane_group = 19
            elif lane_id == "1112_0":
                lane_group = 20
            elif lane_id == "1112_1":
                lane_group = 21
            elif lane_id == "1312_0":
                lane_group = 22
            elif lane_id == "1312_1":
                lane_group = 23
            elif lane_id == "1222_0":
                lane_group = 24
            elif lane_id == "1222_1":
                lane_group = 25
            elif lane_id == "3222_0":
                lane_group = 26
            elif lane_id == "3222_1":
                lane_group = 27
            elif lane_id == "2122_0":
                lane_group = 28
            elif lane_id == "2122_1":
                lane_group = 29
            elif lane_id == "2322_0":
                lane_group = 30
            elif lane_id == "2322_1":
                lane_group = 31
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 31:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-319
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state


    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

