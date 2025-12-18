import bt_library as btl


class ChargingBattery(btl.Task):
    """
    Implementation of the Task "Charge battery".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Charging the battery")

        return self.report_succeeded(blackboard)