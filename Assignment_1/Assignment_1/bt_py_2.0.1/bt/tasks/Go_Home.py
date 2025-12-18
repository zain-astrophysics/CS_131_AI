import bt_library as btl
from ..globals import HOME_PATH

class GoHome(btl.Task):
    """
    Implementation of the Task "Find Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Go Home")

        return self.report_succeeded(blackboard)