#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# version 2.0.1 - copyright (c) 2023-2024 Santini Fabrizio. All rights reserved.
#

# Importing essential libraries 
import bt_library as btl
from bt.behavior_tree import *
from bt.globals import *


# User prompt to select base or experimental version
print(" *** Choose the version to run ***")
print('1. Base Version ')
print('2. Experimental Version ')


choice = input("Enter 1 or 2: ")


# Main body of the assignment
# Initizliaing the variables on blackboard before running the BT 
current_blackboard = btl.Blackboard()
current_blackboard.set_in_environment(BATTERY_LEVEL,50)
current_blackboard.set_in_environment(SPOT_CLEANING, True)
current_blackboard.set_in_environment(GENERAL_CLEANING, True)
current_blackboard.set_in_environment(DUSTY_SPOT_SENSOR,False)
current_blackboard.set_in_environment(HOME_PATH, "")
current_blackboard.set_in_environment(CLEAN_SPOT, True)
current_blackboard.set_in_environment(CLEAN_FLOOR, 0.3)
current_blackboard.set_in_environment(DO_NOTHING, True)
current_blackboard.set_in_environment(AI_PET_SCAN, True)
current_blackboard.set_in_environment(PET_LOCATION, True)


# Selecting base or experimental version
if choice == "1":
    print("Running Base Version ...")
    tree_root = base_version(current_blackboard)
elif choice == "2":
    print("Running Experimental Version ...")
    tree_root = experimental_version(current_blackboard)
else:
    print("Invalid choice. Please enter 1 or 2")

# Initializing battery varibale that will be updated
done = False
battery = current_blackboard.get_in_environment(BATTERY_LEVEL, 0)


while not done:
#     # Each cycle in this while-loop is equivalent to 1 second time
    battery = current_blackboard.get_in_environment(BATTERY_LEVEL, 0)
    spot_cleaning = current_blackboard.get_in_environment(SPOT_CLEANING, False)
    general_cleaning = current_blackboard.get_in_environment(GENERAL_CLEANING, False)
    dusty_sensor = current_blackboard.get_in_environment(DUSTY_SPOT_SENSOR, False)
    clean_floor = current_blackboard.get_in_environment(CLEAN_FLOOR, 0)
    petCheck = current_blackboard.get_in_environment(AI_PET_SCAN, False)
    locationCheck = current_blackboard.get_in_environment(PET_LOCATION, False)
        
    if battery < 30:    
        print(" ***  Battery is low *** ", battery)
        # current_blackboard.set_in_environment(HOME_PATH, "DOCKING")
        # current_blackboard.set_in_environment(BATTERY_LEVEL, battery)
        current_blackboard.set_in_environment(SPOT_CLEANING, False)
        current_blackboard.set_in_environment(GENERAL_CLEANING, False)

    else:
        battery = max(0, battery - 2)
        current_blackboard.set_in_environment(HOME_PATH, "")
        current_blackboard.set_in_environment(SPOT_CLEANING, spot_cleaning)
        current_blackboard.set_in_environment(DUSTY_SPOT_SENSOR, dusty_sensor)
        current_blackboard.set_in_environment(CLEAN_FLOOR, clean_floor)
        current_blackboard.set_in_environment(CLEAN_SPOT, False)
        current_blackboard.set_in_environment(GENERAL_CLEANING, general_cleaning)
        current_blackboard.set_in_environment(AI_PET_SCAN, petCheck)
        current_blackboard.set_in_environment(PET_LOCATION, locationCheck)

    # Updating battery level at every loop
    current_blackboard.set_in_environment(BATTERY_LEVEL, battery)    

    # running the BT
    result = tree_root.run(current_blackboard)
    spot = current_blackboard.get_in_environment(SPOT_CLEANING, False)
    general = current_blackboard.get_in_environment(GENERAL_CLEANING, False)

    # Check if all tasks are complete
    if not spot and not general and result == btl.ResultEnum.SUCCEEDED:
        print("All tasks completed. Shutting down.")
        done = True


