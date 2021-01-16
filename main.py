from constants import *
from Simulator1 import Simulator1
from Simulator2 import Simulator2

if __name__ == '__main__':

    if MODEL == 1:
        simulator = Simulator1()
    else:
        simulator = Simulator2()

    simulator.simulate ()
