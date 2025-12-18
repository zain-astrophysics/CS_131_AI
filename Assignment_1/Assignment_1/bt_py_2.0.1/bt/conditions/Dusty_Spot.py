import bt_library as btl
from ..globals import DUSTY_SPOT_SENSOR

class DustySpot(btl.Condition):
    """
    Implementation of the condition "Dusty Spot.
    """ 
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Checking Dusty Spot")

        dusty_spot_detected = blackboard.get_in_environment(DUSTY_SPOT_SENSOR, False)

        if dusty_spot_detected:
            return self.report_succeeded(blackboard)
        else:
            return self.report_failed(blackboard)