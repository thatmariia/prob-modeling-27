from enum import IntEnum, Enum

SIMULATION_EPOCHS = 100000
MODEL = 2

"""--- MODEL 1 ---"""

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
    GOOD_MIN        = -45
    GOOD_MAX        =  45
    POSSIBLE_MIN    = -180
    POSSIBLE_MAX    =  180

class LinealAngle(IntEnum):
    """
    Minimum and maximum lateral angle in degrees
    """
    GOOD_MIN        = -45
    GOOD_MAX        =  45
    POSSIBLE_MIN    = -180
    POSSIBLE_MAX    =  180

class FootAngle(IntEnum):
    """
    Minimum and maximum lateral angle in degrees
    """
    GOOD_MIN        = -20
    GOOD_MAX        =  20
    POSSIBLE_MIN    = -120
    POSSIBLE_MAX    =  120


class Outcome1(Enum):
    FAIL            = "FAIL"
    SUCCESS         = "SUCCESS"
    P1              = "P1"
    P2              = "P2"
    UNDETERMINED    = "UNDETERMINED"