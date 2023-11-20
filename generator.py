import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/2_2.rou.xml", "w") as routes:
            print("""<routes>
            <vType id="CarA" length="5" maxSpeed="22" carFollowModel="IDM" actionStepLength="1" tau="1.4" speedDev="0.0" accel="1" decel="2" speedFactor="1.0" minGap="2" delta="4" stepping="1.5"/>

            <route id="01_11_21_31" edges="0111 1121 2131" />
            <route id="01_11_21_22_32" edges="0111 1121 2122 2232" />
            <route id="01_11_12_22_32" edges="0111 1112 1222 2232" />
            <route id="01_11_12_13" edges="0111 1112 1213" />
            <route id="01_11_21_22_23" edges="0111 1121 2122 2223" />
            <route id="01_11_12_22_23" edges="0111 1112 1222 2223" />

            <route id="02_12_22_21_31" edges="0212 1222 2221 2131" />
            <route id="02_12_11_21_31" edges="0212 1211 1121 2131" />
            <route id="02_12_22_32" edges="0212 1222 2232" />
            <route id="02_12_13" edges="0212 1213" />
            <route id="02_12_22_23" edges="0212 1222 2223" />

            <route id="10_11_21_31" edges="1011 1121 2131" />
            <route id="10_11_21_22_32" edges="1011 1121 2122 2232" />
            <route id="10_11_12_22_32" edges="1011 1112 1222 2232" />
            <route id="10_11_12_13" edges="1011 1112 1213" />
            <route id="10_11_21_22_23" edges="1011 1121 2122 2223" />
            <route id="10_11_12_22_23" edges="1011 1112 1222 2223" />

            <route id="20_21_31" edges="2021 2131" />
            <route id="20_21_22_32" edges="2021 2122 2232" />
            <route id="20_21_11_12_13" edges="2021 2111 1112 1213" />
            <route id="20_21_22_12_13" edges="2021 2122 2212 1213" />
            <route id="20_21_22_23" edges="2021 2122 2223" />""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.7:  # choose direction: straight or turn - 70% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="01_31_%i" type="CarA" route="01_11_21_31" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="02_32_%i" type="CarA" route="02_12_22_32" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="10_13_%i" type="CarA" route="10_11_12_13" depart="%i" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="20_23_%i" type="CarA" route="20_21_22_23" depart="%i" />' % (car_counter, step), file=routes)
                else:  # car that turn -30% of the time the car turns
                    route_turn = np.random.randint(1, 13)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="01_32_1_%i" type="CarA" route="01_11_21_22_32" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="01_13_%i" type="CarA" route="01_11_12_13" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="01_23_1_%i" type="CarA" route="01_11_21_22_23" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="02_31_1_%i" type="CarA" route="02_12_22_21_31" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="02_13_%i" type="CarA" route="02_12_13" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="02_23_%i" type="CarA" route="02_12_22_23" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="10_31_%i" type="CarA" route="10_11_21_31" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="10_32_1_%i" type="CarA" route="10_11_21_22_32" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 9:
                        print('    <vehicle id="10_23_1_%i" type="CarA" route="10_11_21_22_23" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 10:
                        print('    <vehicle id="20_31_%i" type="CarA" route="20_21_31" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 11:
                        print('    <vehicle id="20_32_%i" type="CarA" route="20_21_22_32" depart="%i" />' % (car_counter, step), file=routes)
                    elif route_turn == 12:
                        print('    <vehicle id="20_13_1_%i" type="CarA" route="20_21_11_12_13" depart="%i" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)
