from constants import *

import numpy as np
import matplotlib.pyplot as plt
import collections

class Simulator1:

    def __init__(self):
        self.outcomes = []

    def simulate(self):

        for _ in range(SIMULATION_EPOCHS):

            # since the person is inexperienced, randomly draw the values from the uniform dist of all 'physically' possible
            stepSize        = self._drawUniform(low=StepSize.POSSIBLE_MIN.value, high=StepSize.POSSIBLE_MAX.value)
            lateralAngle    = self._drawUniform(low=LateralAngle.POSSIBLE_MIN.value, high=LateralAngle.POSSIBLE_MAX.value)

            outcome = self._determineOutcome(stepSize=stepSize, lateralAngle=lateralAngle)
            self.outcomes.append(outcome)

        self._plot_outcomes()

    def _plot_outcomes(self):
        size = 20
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

        fig, ax = plt.subplots (1, figsize=(20, 10))

        frequencies = collections.Counter(self.outcomes)

        ax.bar(frequencies.keys(), frequencies.values(), width=0.8, color=colors)
        plt.xticks(np.arange(len(frequencies)), frequencies.keys())
        ax.set_title("Outcomes in {} tries, Model 1".format(SIMULATION_EPOCHS))
        ax.set_xlabel("Outcome1")
        ax.set_ylabel("Frequency")

        fig.show()


    def _determineOutcome(self, stepSize, lateralAngle):

        if ((stepSize < StepSize.GOOD_MIN.value) or (stepSize > StepSize.GOOD_MAX.value)) and \
           ((lateralAngle < LateralAngle.GOOD_MIN.value) or (lateralAngle > LateralAngle.GOOD_MAX.value)):
            return Outcome1.FAIL.value

        if (stepSize >= StepSize.GOOD_MIN.value) and (stepSize <= StepSize.GOOD_MAX.value) and \
           ((lateralAngle < LateralAngle.GOOD_MIN.value) or (lateralAngle > LateralAngle.GOOD_MAX.value)):
            return Outcome1.P1.value

        if (lateralAngle >= LateralAngle.GOOD_MIN.value) and (lateralAngle <= LateralAngle.GOOD_MAX.value) and \
           ((stepSize < StepSize.GOOD_MIN.value) or (stepSize > StepSize.GOOD_MAX.value)):
            return Outcome1.P2.value

        if ((stepSize >= StepSize.GOOD_MIN.value) and (stepSize <= StepSize.GOOD_MAX.value)) and \
           ((lateralAngle >= LateralAngle.GOOD_MIN.value) or (lateralAngle <= LateralAngle.GOOD_MAX.value)):
            return Outcome1.SUCCESS.value

        return Outcome1.UNDETERMINED


    def _drawUniform(self, low, high):
        return np.random.uniform(low=low, high=high+1)