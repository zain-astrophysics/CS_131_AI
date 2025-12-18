
import bt_library as btl
from ..globals import  PET_LOCATION, AI_PET_SCAN
class DoneCleanFloor(btl.Task):
    """
    Implementation of the Task "Done Clean floor".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message(" *** Done Cleaning Floor *** ")
        blackboard.set_in_environment(PET_LOCATION,False)
        blackboard.set_in_environment(AI_PET_SCAN,False)

        return self.report_failed(blackboard)