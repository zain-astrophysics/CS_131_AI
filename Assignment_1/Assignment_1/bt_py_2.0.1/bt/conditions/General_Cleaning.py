import bt_library as btl
from ..globals import GENERAL_CLEANING, CLEAN_SPOT

class GeneralCleaning(btl.Condition):
    """
    Implementation of the condition "Spot Cleaning.
    """ 
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Checking General Cleaning")
        current_state = blackboard.get_in_environment(GENERAL_CLEANING, False)
        clean_spot = blackboard.get_in_environment(CLEAN_SPOT, False)
        if current_state:
            print(f"Current State: {current_state}")
            return self.report_succeeded(blackboard)
        else:
            print(f"General Cleaning is inactive. Current State: {current_state}")
            return self.report_failed(blackboard)