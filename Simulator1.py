from constants import *

import numpy as np
import matplotlib.pyplot as plt
import collections
from enum import IntEnum, Enum

class StepSize(IntEnum):
    """
    Minimum and maximum step size in centimeters
    """
    GOOD_MIN        = 72
    GOOD_MAX        = 104
    POSSIBLE_MIN    = 0
    POSSIBLE_MAX    = 160

class LateralAngle(IntEnum):
    """
    Minimum and maximum lateral angle in degrees
    """
    GOOD_MIN        = -30
    GOOD_MAX        =  30
    POSSIBLE_MIN    = -90
    POSSIBLE_MAX    =  90

class Outcome(Enum):
    FAIL            = "FAIL"
    SUCCESS         = "SUCCESS"
    P1              = "P1"
    P2              = "P2"
    UNDETERMINED    = "UNDETERMINED"

class Simulator1:

    def __init__(self):
        self.probHistory = []
        self.paramHistory = []

    """---SIMULATING---"""

    def simulate(self):

        for _ in range(SIMULATION_EPOCHS):

            # since the person is inexperienced, randomly draw the values from the uniform dist of all 'physically' possible
            stepSize        = self._drawUniform(low=StepSize.POSSIBLE_MIN.value, high=StepSize.POSSIBLE_MAX.value)
            lateralAngle    = self._drawUniform(low=LateralAngle.POSSIBLE_MIN.value, high=LateralAngle.POSSIBLE_MAX.value)

            outcome = self._determineOutcome(stepSize=stepSize, lateralAngle=lateralAngle)
            self.probHistory.append(outcome)

            params = {
                    "Step size"     : stepSize,
                    "Lateral angle" : lateralAngle
            }
            self.paramHistory.append(params)

        # UNCOMMENT TO PLOT
        #self._plotOutcomes()
        #self._plotParams()

    def _determineOutcome(self, stepSize, lateralAngle):

        if ((stepSize < StepSize.GOOD_MIN.value) or (stepSize > StepSize.GOOD_MAX.value)) and \
           ((lateralAngle < LateralAngle.GOOD_MIN.value) or (lateralAngle > LateralAngle.GOOD_MAX.value)):
            return Outcome.FAIL.value

        if (stepSize >= StepSize.GOOD_MIN.value) and (stepSize <= StepSize.GOOD_MAX.value) and \
           ((lateralAngle < LateralAngle.GOOD_MIN.value) or (lateralAngle > LateralAngle.GOOD_MAX.value)):
            return Outcome.P1.value

        if (lateralAngle >= LateralAngle.GOOD_MIN.value) and (lateralAngle <= LateralAngle.GOOD_MAX.value) and \
           ((stepSize < StepSize.GOOD_MIN.value) or (stepSize > StepSize.GOOD_MAX.value)):
            return Outcome.P2.value

        if ((stepSize >= StepSize.GOOD_MIN.value) and (stepSize <= StepSize.GOOD_MAX.value)) and \
           ((lateralAngle >= LateralAngle.GOOD_MIN.value) or (lateralAngle <= LateralAngle.GOOD_MAX.value)):
            return Outcome.SUCCESS.value

        return Outcome.UNDETERMINED


    def _drawUniform(self, low, high):
        return np.random.uniform(low=low, high=high+1)

    """---PLOTTING---"""

    def _plotParams(self):
        size = 30
        plot_params = {
                'legend.fontsize': 'large',
                'figure.figsize' : (20, 8),
                'axes.labelsize' : size,
                'axes.titlesize' : size * 2,
                'xtick.labelsize': size,
                'ytick.labelsize': size,
                'axes.titlepad'  : 25
        }
        plt.rcParams.update(plot_params)
        colors = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a", "#f4a261"]

        params = self.__extractParams()

        c = 0
        for (param, values) in params.items():
            fig, ax = plt.subplots(1, figsize=(20, 10))

            ax.scatter(self.probHistory, values, c=colors[c % len(colors)])
            c += 1

            #plt.xticks (np.arange (len (frequencies)), frequencies.keys ())
            ax.set_title("{} in {} epochs, Model 1".format(param, SIMULATION_EPOCHS))
            ax.set_xlabel("Outcome")
            ax.set_ylabel(param)

            fig.show()


    def __extractParams(self):
        params = {}

        for i in range(SIMULATION_EPOCHS):
            for (param, value) in self.paramHistory[i].items():

                if not (param in params):
                    params[param] = []

                params[param].append(value)

        return params



    def _plotOutcomes(self):
        size = 30
        plot_params = {
                'legend.fontsize': 'large',
                'figure.figsize' : (20, 8),
                'axes.labelsize' : size,
                'axes.titlesize' : size*2,
                'xtick.labelsize': size,
                'ytick.labelsize': size,
                'axes.titlepad'  : 25
        }
        plt.rcParams.update(plot_params)
        colors = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a", "#f4a261"]

        fig, ax = plt.subplots(1, figsize=(20, 10))

        frequencies = collections.Counter(self.probHistory)
        print(frequencies)

        ax.bar(frequencies.keys(), frequencies.values(), width=0.8, color=colors)
        plt.xticks(np.arange(len(frequencies)), frequencies.keys())
        ax.set_title("Outcomes in {} epochs, Model 1".format(SIMULATION_EPOCHS))
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Frequency")

        fig.show()