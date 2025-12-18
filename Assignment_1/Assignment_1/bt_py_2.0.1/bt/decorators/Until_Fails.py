# btl_library/decorator.py


import bt_library as btl


class UntilFails(btl.Decorator):
    """
    Implementation of the condition "Run Clean Spot Check".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        """
        Run the child behavior until it fails.

        :param blackboard: The blackboard holding environment information
        :return: ResultEnum (SUCCEEDED if the child fails, RUNNING otherwise)
        """
        self.print_message("Running until clean floor fails...")
        
        # Check the condition (if CLEAN_FLOOR is False)
        # clean_floor = blackboard.get_in_environment(CLEAN_FLOOR, True)
        
        # # If CLEAN_FLOOR is False, return a failed state
        # if not clean_floor:
        #     return btl.ResultEnum.FAILED
        
        # Run the child behavior
        result = self.child.run(blackboard)
        
        # If the child behavior fails, return SUCCEEDED
        if result == btl.ResultEnum.FAILED:
            print('*** Finished Cleaning Floor ***')
            return btl.ResultEnum.SUCCEEDED
        else:
        # Otherwise, return RUNNING (if the child is still running or succeeded)
            return btl.ResultEnum.RUNNING


