
import bt_library as btl
from ..globals import  SPOT_CLEANING

class DoneSpot(btl.Task):
    """
    Implementation of the Task "Find Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Done Spot")
        blackboard.set_in_environment(SPOT_CLEANING, False)

        return self.report_succeeded(blackboard)