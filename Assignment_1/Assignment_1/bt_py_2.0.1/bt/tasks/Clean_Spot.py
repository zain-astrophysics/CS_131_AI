
import bt_library as btl


class CleanSpot(btl.Task):
    """
    Implementation of the Task "Clean Spot".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum: 
            # if not blackboard.get_in_environment(CLEAN_SPOT, False):
                # return self.report_failed(blackboard)

            self.print_message("Clean Spot")
            return self.report_succeeded(blackboard)

        # return self.report_succeeded(blackboard) \
        #     if blackboard.get_in_environment(CLEAN_SPOT, False) \
        #     else self.report_failed(blackboard)





        # return self.report_succeeded(blackboard)