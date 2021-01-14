from constants import *

from math import sqrt, pi
import numpy as np
from statistics import NormalDist, mean
import matplotlib.pyplot as plt
import collections

HEIGHT = 170
FITNESS = 5
T = 110
MANUAL_WEIGHT = 180


class Simulator2:

    def __init__(self):
        self.maxSteps = 20 * FITNESS

        self.stepCount      = 0
        self.extraWeight    = None
        self.energyLevel    = None
        self.stepSize       = None
        self.footAngle      = None
        self.lateralAngle   = None
        self.linealAngle    = None

        self.probHistory = {} # epoch : array of probs
        self.paramsHistory = {} # epoch : [param : val]

    def simulate(self):
        for epoch in range(SIMULATION_EPOCHS):
            self._setupEpoch()
            self._performLunges(epoch=epoch)

        #self.plotAvgHistory()
        self.plotFailinPoints()
        #self.plotParamEvolutions()

    """---PLOTTING---"""

    def plotParamEvolutions(self):
        size = 40
        plot_params = {
                'legend.fontsize' : 'large',
                'figure.figsize'  : (20, 8),
                'axes.labelsize'  : size * 1.2,
                'axes.titlesize'  : size * 1.7,
                'xtick.labelsize' : size * 1.2,
                'ytick.labelsize' : size * 1.2,
                'axes.titlepad'   : 25,
                'lines.markersize': size
        }
        plt.rcParams.update (plot_params)
        colors = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a", "#f4a261"]
        red = "#FF0000"

        avgParamEvolution = self._extractAvgParamsEvolution()

        c = 0
        for (param, evol) in avgParamEvolution.items():
            fig, ax = plt.subplots(1, figsize=(25, 13))

            x = list(range(1, len(evol) + 1))

            ax.scatter(x, evol, c=colors[c % len(colors)])
            c += 1
            """
            minGood, maxGood = self.__getGoodRange(param=param)
            y_minGood = [minGood for _ in range(1, len(evol) + 1)]
            y_maxGood = [maxGood for _ in range(1, len(evol) + 1)]

            ax.plot(y_minGood, c=red, lw=5)
            ax.plot(y_maxGood, c=red, lw=5)
            """
            #ax.set_xlim([0, T-1])
            #plt.xticks(np.arange(1, len(evol) + 1, 5))
            ax.set_title("Avg {} evol in {} epochs, Model 2\nl={}, h={}, w={}".format (param, SIMULATION_EPOCHS, FITNESS, HEIGHT, MANUAL_WEIGHT))
            ax.set_xlabel("t")
            ax.set_ylabel("Avg value")

            fig.show()

    def __getGoodRange(self, param):
        if param == "Extra weight":
            return 0, 10 * FITNESS
        if param == "Energy level":
            return 0.1, 1
        if param == "Step size":
            return 0.45 * HEIGHT, 0.65 * HEIGHT
        if param == "Foot angle":
            return -60, 60
        if param == "Lateral angle":
            w_max_p = 105 * FITNESS
            a_min = -12*pi * sqrt(self.extraWeight / w_max_p)
            a_max = -1 * a_min
            return a_min, a_max
        if param == "Lineal angle":
            w_max_p = 105 * FITNESS
            b_min = -12 * pi * sqrt (self.extraWeight / w_max_p)
            b_max = -1 * b_min
            return b_min, b_max
        if param == "Probability":
            return 0, 1

    def plotFailinPoints(self):
        size = 40
        plot_params = {
                'legend.fontsize' : 'large',
                'figure.figsize'  : (20, 8),
                'axes.labelsize'  : size * 1.2,
                'axes.titlesize'  : size * 1.7,
                'xtick.labelsize' : size * 1.2,
                'ytick.labelsize' : size * 1.2,
                'axes.titlepad'   : 25,
                'lines.markersize': size
        }
        plt.rcParams.update (plot_params)
        colors = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a", "#f4a261"]

        fig, ax = plt.subplots (1, figsize=(25, 13))

        failingPoints = self.__computeFailingPoints()
        print(failingPoints)

        ax.bar(failingPoints.keys(), failingPoints.values(), width=0.8, color=colors)
        plt.xticks(np.arange(1, T+1, 5))
        ax.set_title("Failing points, {} epochs, Model 2\nl={}, h={}, w={}".format (SIMULATION_EPOCHS, FITNESS, HEIGHT, MANUAL_WEIGHT))
        ax.set_xlabel("t")
        ax.set_ylabel("Fail frequency")

        fig.show()


    def plotAvgHistory(self):
        size = 40
        plot_params = {
                'legend.fontsize' : 'large',
                'figure.figsize'  : (20, 8),
                'axes.labelsize'  : size * 1.2,
                'axes.titlesize'  : size * 1.7,
                'xtick.labelsize' : size * 1.2,
                'ytick.labelsize' : size * 1.2,
                'axes.titlepad'   : 25,
                'lines.markersize': size
        }
        plt.rcParams.update (plot_params)
        colors = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a", "#f4a261"]

        fig, ax = plt.subplots (1, figsize=(20, 20))

        avgHistory = self.__computeAvgHistory()
        x = list(range(1, T+1))

        ax.scatter(x, avgHistory, c=colors[4])
        ax.set_title("Avg probs in {} epochs, Model 2\nl={}, h={}".format(SIMULATION_EPOCHS, FITNESS, HEIGHT))
        ax.set_xlabel("t")
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])

        fig.show()

    def _extractAvgParamsEvolution(self):
        """
        :return: {t : {param : avg value}} --> {param : [avg values]}
        """
        avgParamEvolutionPerStep = self._computeAvgParamEvolution ()
        avgParamEvolution = {}

        for (t, avgParams) in avgParamEvolutionPerStep.items ():
            for (param, avgValue) in avgParams.items ():

                if not (param in avgParamEvolution):
                    avgParamEvolution[param] = []

                avgParamEvolution[param].append (avgValue)

        return avgParamEvolution

    def _computeAvgParamEvolution(self):

        paramEvolution = {}

        for (epoch, setParams) in self.paramsHistory.items():
            for t in range(len(setParams)):

                if not (t in paramEvolution):
                    paramEvolution[t] = {}

                for (param, value) in setParams[t].items():

                    if param in paramEvolution[t]:
                        paramEvolution[t][param].append(value)
                    else:
                        paramEvolution[t][param] = [value]

        avgParamEvolution = {}

        for (t, params) in paramEvolution.items():

            if not (t in avgParamEvolution):
                avgParamEvolution[t] = {}

            for (param, values) in paramEvolution[t].items():
                avgParamEvolution[t][param] = mean(values)

        return avgParamEvolution

    def __computeFailingPoints(self):
        fails = np.zeros(shape=T)

        for (_, probs) in self.probHistory.items():
            if 0.0 in probs:
                fails[probs.index(0.0)] += 1

        failingPoints = {}

        for t in range(T):
            if fails[t] > 0:
                failingPoints[t+1] = fails[t]

        return failingPoints

    def __computeAvgHistory(self):
        stepHistory = self.__computeStepHistory()
        return [mean(stepHistory[i]) for i in range(T)]

    def __computeStepHistory(self):
        stepHistory = np.zeros(shape=(T, SIMULATION_EPOCHS))

        for (epoch, probs) in self.probHistory.items ():
            for i in range(T):
                stepHistory[i][epoch] = probs[i]

        return stepHistory

    """---SIMULATING---"""

    def _setupEpoch(self):
        self.extraWeight = self._updateExtraWeight() if (MANUAL_WEIGHT == None) else MANUAL_WEIGHT

    def _performLunges(self, epoch):
        successProbabilities = []
        parameterValues = []
        failed = False

        for currStep in range(T):
            self.stepCount      = currStep
            self.energyLevel    = self._updateEnergyLevel()
            self.stepSize       = self._updateStepSize()
            self.footAngle      = self._updateFootAngle()
            self.lateralAngle   = self._updateLateralAngle()
            self.linealAngle    = self._updateLinealAngle()

            successProbability  = self._computeSuccessProbability()
            successProbabilities.append(successProbability)

            currParameterValues = self.__getCurrParameterValues(probability=successProbability)
            parameterValues.append(currParameterValues)

            if successProbability == 0.0:
                l = T - len(successProbabilities)
                successProbabilities += [0 for _ in range(l)]
                break

        self.probHistory[epoch] = successProbabilities
        self.paramsHistory[epoch] = parameterValues

    def __getCurrParameterValues(self, probability):
        currParameterValues = {
                "Extra weight"  : self.extraWeight,
                "Energy level"  : self.energyLevel,
                "Step size"     : self.stepSize,
                "Foot angle"    : self.footAngle,
                "Lateral angle" : self.lateralAngle,
                "Lineal angle"  : self.linealAngle,
                "Probability"   : probability
        }
        return  currParameterValues

    """---PROBABILITIES---"""

    def _computeSuccessProbability(self):
        if self._isDefiniteFail():
            return 0.0
        if self._isDefiniteSuccess():
            return 1.0
        return self._probabilityOtherwise()

    def _probabilityOtherwise(self):
        lw = 1.0 / 5.0
        Qw = self.__Qw()

        ls = 1.0 / 6.0
        Qs = self.__Qs()

        lg = 1.0 / 6.0
        Qg = self.__Qg()

        la = 7.0 / 30.0
        Qa = self.__Qa()

        lb = 7.0 / 30.0
        Qb = self.__Qb()

        return lw*Qw + ls*Qs + lg*Qg + la*Qa + lb*Qb

    def __Qw(self):
        w_min   = 0.0
        w_max   = 10.0 * FITNESS

        w_min_p = 0.0
        w_max_p = 105.0 * FITNESS

        W_good  = abs(w_max - w_min)
        range   = abs(w_max + (w_max_p-w_max) * ((1.0-0.2*FITNESS)**3))

        return W_good / range

    def __Qs(self):
        s_min = 0.45 * HEIGHT
        s_max = 0.65 * HEIGHT

        avg = 0.5 * HEIGHT
        std = (0.5 * HEIGHT) * sqrt (1.0 / self.energyLevel)

        return self.__cdfRange(avg=avg, std=std, high=s_max, low=s_min)

    def __Qg(self):
        g_min = -60.0
        g_max = 60.0

        avg = 0.0
        std = (g_max / FITNESS) * sqrt (1.0 / self.energyLevel)

        return self.__cdfRange (avg=avg, std=std, high=g_max, low=g_min)

    def __Qa(self):
        a_min = -60.0
        a_max = 60.0

        w_max_p = 105.0 * FITNESS

        avg = 0.0
        std = (12.0*pi / FITNESS) * sqrt(self.extraWeight / w_max_p) * sqrt (1.0 / self.energyLevel)

        return self.__cdfRange (avg=avg, std=std, high=a_max, low=a_min)

    def __Qb(self):
        b_min = -90.0
        b_max = 90.0

        w_max_p = 105.0 * FITNESS

        avg = 0
        std = (12.0 * pi / FITNESS) * sqrt (self.extraWeight / w_max_p) * sqrt (1.0 / self.energyLevel)

        return self.__cdfRange (avg=avg, std=std, high=b_max, low=b_min)

    def __cdfRange(self, avg, std, high, low):
        phi = NormalDist(mu=avg, sigma=std)
        return phi.cdf(high) - phi.cdf(low)

    def _isDefiniteFail(self):
        if not self.__isGoodEnergyLevel():
            return True
        if not self.__isGoodStepCount():
            return True
        if self.__isGoodExtraWeight():
            return False
        if self.__isGoodStepSize():
            return False
        if self.__isGoodFootAngle():
            return False
        if self.__isGoodLateralAngle():
            return False
        if self.__isGoodLinealAngle():
            return False
        return True

    def _isDefiniteSuccess(self):
        if not self.__isGoodStepCount():
            return False
        if not self.__isGoodExtraWeight():
            return False
        if not self.__isGoodEnergyLevel():
            return False
        if not self.__isGoodStepSize():
            return False
        if not self.__isGoodFootAngle():
            return False
        if not self.__isGoodLateralAngle():
            return False
        if not self.__isGoodLinealAngle():
            return False
        return True

    """---RANGE CHECKING---"""

    def __isGoodStepCount(self):
        return self.stepCount <= self.maxSteps

    def __isGoodExtraWeight(self):
        w_min = 0.0
        w_max = 10.0 * FITNESS
        return (self.extraWeight >= w_min) and (self.extraWeight <= w_max)

    def __isGoodEnergyLevel(self):
        e_min = 0.1
        return self.energyLevel >= e_min

    def __isGoodStepSize(self):
        s_min = 0.45 * HEIGHT
        s_max = 0.65 * HEIGHT
        return (self.stepSize >= s_min) and (self.stepSize <= s_max)

    def __isGoodFootAngle(self):
        g_min = -60.0
        g_max = 60.0
        return (self.footAngle >= g_min) and (self.footAngle <= g_max)

    def __isGoodLateralAngle(self):
        w_max_p = 105.0 * FITNESS
        a_min   = -12.0 * pi * sqrt(self.extraWeight / w_max_p)
        a_max   =  12.0 * pi * sqrt (self.extraWeight / w_max_p)
        return (self.lateralAngle >= a_min) and (self.lateralAngle <= a_max)

    def __isGoodLinealAngle(self):
        w_max_p = 105.0 * FITNESS
        b_min   = -12.0 * pi * sqrt(self.extraWeight / w_max_p)
        b_max   =  12.0 * pi * sqrt(self.extraWeight / w_max_p)
        return (self.linealAngle >= b_min) and (self.linealAngle <= b_max)

    """---UPDATING---"""

    def _updateEnergyLevel(self):
        maxWeight = self.__getMaxWeight()
        return 1.0 - sqrt((self.stepCount / self.maxSteps) * (self.extraWeight / maxWeight))

    def _updateExtraWeight(self):
        w_min_p = 0.0

        minWeight = w_min_p
        maxWeight = self.__getMaxWeight()

        return np.random.uniform(low=minWeight, high=maxWeight+1)

    def __getMaxWeight(self):
        w_max   = 10.0 * FITNESS
        w_max_p = 105.0 * FITNESS

        p_w     = 0.2 * FITNESS

        return w_max + (w_max_p - w_max) * ((1.0 - p_w) ** 3)

    def _updateStepSize(self):
        avg = 0.5 * HEIGHT
        std = (0.5 * HEIGHT / FITNESS) * sqrt(1.0 / self.energyLevel)

        s_min_p = 0.0
        s_max_p = HEIGHT

        ss = np.random.normal(loc=avg, scale=std)
        while (ss < s_min_p) or (ss > s_max_p):
            ss = np.random.normal(loc=avg, scale=std)

        return ss

    def _updateFootAngle(self):
        g_max   =  60.0

        g_min_p = -180.0
        g_max_p =  180.0

        avg = 0.0
        std = (g_max / FITNESS) * sqrt(1.0 / self.energyLevel)

        fa = np.random.normal(loc=avg, scale=std)
        while (fa < g_min_p) or (fa > g_max_p):
            fa = np.random.normal (loc=avg, scale=std)

        return fa

    def _updateLateralAngle(self):
        a_min_p = -90.0
        a_max_p =  90.0

        avg = 0.0
        std = ((12.0 * pi / FITNESS) * sqrt(self.extraWeight / self.__getMaxWeight())) * sqrt(1.0 / self.energyLevel)

        la = np.random.normal(loc=avg, scale=std)
        while (la < a_min_p) or (la > a_max_p):
            la = np.random.normal(loc=avg, scale=std)

        return la

    def _updateLinealAngle(self):
        b_min_p = -60.0
        b_max_p = 60.0

        avg = 0.0
        std = ((12.0 * pi / FITNESS) * sqrt (self.extraWeight / self.__getMaxWeight ())) * sqrt(1.0 / self.energyLevel)

        la = np.random.normal(loc=avg, scale=std)
        while (la < b_min_p) or (la > b_max_p):
            la = np.random.normal(loc=avg, scale=std)

        return la

    """---MISC---"""

    def printValues(self):
        print("----------------------------------")
        print("Step count    = ", self.stepCount)
        print("Extra weight  = ", self.extraWeight)
        print("Energy level  = ", self.energyLevel)
        print("Step size     = ", self.stepSize)
        print("Foot angle    = ", self.footAngle)
        print("Lateral angle = ", self.lateralAngle)
        print("Lineal angle  = ", self.linealAngle)
        print ("----------------------------------")
