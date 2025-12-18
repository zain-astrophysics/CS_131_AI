
import bt_library as btl
from ..globals import CLEAN_FLOOR
import random

class CleanFloor(btl.Task):
    """
    Implementation of the Task "Clear Floor".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Cleaning Floor")
        percent = blackboard.get_in_environment(CLEAN_FLOOR, 0)
        if random.random() < percent:
            return self.report_failed(blackboard)
        return self.report_succeeded(blackboard)