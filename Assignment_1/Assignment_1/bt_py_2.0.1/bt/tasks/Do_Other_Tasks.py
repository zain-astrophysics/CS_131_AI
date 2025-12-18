import bt_library as btl
from ..globals import CLEAN_FLOOR

class DoOtherTasks(btl.Task):
    """
    Implementation of the Task "Do Other Tasks".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Pet location is not empty....continuing other tasks")


        return self.report_failed(blackboard)