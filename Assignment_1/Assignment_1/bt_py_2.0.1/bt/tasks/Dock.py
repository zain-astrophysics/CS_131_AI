import bt_library as btl


class Docking(btl.Task):
    """
    Implementation of the Task "Find Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Docking")

        return self.report_succeeded(blackboard)