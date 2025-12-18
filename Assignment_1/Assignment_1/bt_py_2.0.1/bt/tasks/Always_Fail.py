
import bt_library as btl

class AlwaysFail(btl.Task):
    """
    Implementation of the Task "Find Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Always Fail")

        return self.report_failed(blackboard)