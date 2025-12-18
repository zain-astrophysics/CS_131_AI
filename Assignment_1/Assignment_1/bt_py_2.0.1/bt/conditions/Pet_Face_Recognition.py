import bt_library as btl
from ..globals import  AI_PET_SCAN

class PetScanRecognition(btl.Condition):
    """
    Implementation of the condition "AI Pet Face Recognition.
    """ 
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Check if Cat or Dog is in house")

        pet_detected = blackboard.get_in_environment(AI_PET_SCAN, False)

        if  pet_detected:
            return self.report_succeeded(blackboard)
        else:
            return self.report_failed(blackboard)