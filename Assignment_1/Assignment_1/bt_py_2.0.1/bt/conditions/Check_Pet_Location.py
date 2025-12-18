import bt_library as btl
from ..globals import PET_LOCATION

class CheckPetLocation(btl.Condition):
    """
    Implementation of the condition "Checking Pet location if its empty.
    """ 
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message(" *** Checking if the Pet location is empty and saving the grid point ***")

        pet_location = blackboard.get_in_environment(PET_LOCATION, False)

        if pet_location:
            return self.report_succeeded(blackboard)
        else:
            return self.report_failed(blackboard)