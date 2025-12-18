# bt/decorators/interrupt_if_low_battery.py

import bt_library as btl

from ..globals import BATTERY_LEVEL

class LowBatteryInterrupt(btl.Decorator):
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        battery = blackboard.get_in_environment(BATTERY_LEVEL, 100)
        
        if battery < 30:
            self.print_message(" *** INTERRUPTED: Battery too low *** ")
            return self.report_failed(blackboard)

        return self.child.run(blackboard)
