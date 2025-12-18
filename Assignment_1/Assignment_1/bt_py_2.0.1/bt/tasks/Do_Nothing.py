
import bt_library as btl
from ..globals import DO_NOTHING
class DoNothing(btl.Task):
    """
    Implementation of the Task "Do Nothing".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Do Nothing")
        blackboard.set_in_environment(DO_NOTHING, False )
        return self.report_succeeded(blackboard)