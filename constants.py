from enum import IntEnum, Enum

SIMULATION_EPOCHS = 10000

class StepSize(IntEnum):
    """
    Minimum and maximum step size in centimeters
    """
    GOOD_MIN        = 1
    GOOD_MAX        = 100
    POSSIBLE_MIN    = 1
    POSSIBLE_MAX    = 200

class LateralAngle(IntEnum):
    """
    Minimum and maximum lateral angle in degrees
    """
    GOOD_MIN        = -30
    GOOD_MAX        =  30
    POSSIBLE_MIN    = -180
    POSSIBLE_MAX    =  180

class Outcome(Enum):
    FAIL            = "FAIL"
    SUCCESS         = "SUCCESS"
    P1              = "P1"
    P2              = "P2"
    UNDETERMINED    = "UNDETERMINED"