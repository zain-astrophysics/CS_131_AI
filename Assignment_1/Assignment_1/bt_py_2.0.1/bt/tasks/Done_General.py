
import bt_library as btl
from ..globals import  DUSTY_SPOT_SENSOR, GENERAL_CLEANING

class DoneGeneral(btl.Task):
    """
    Implementation of the Task "Done General".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("GENERAL CLEANING DONE")
        blackboard.set_in_environment(DUSTY_SPOT_SENSOR,False)
        blackboard.set_in_environment(GENERAL_CLEANING,False)

        return self.report_succeeded(blackboard)