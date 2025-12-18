#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# version 2.0.1 - copyright (c) 2023-2024 Santini Fabrizio. All rights reserved.
#

import bt_library as btl

import bt as bt

# Instantiate the tree according to the assignment. 

# **** BASE TREE ***

def base_version(current_blackboard):
    print('Using base version')
    tree_root = bt.Selection([
        # Priority 1
        bt.Sequence([bt.BatteryLessThan30(),
            bt.FindHome(),
            bt.GoHome(),
            bt.Docking(),
            bt.ChargingBattery() ]),

        # Priority 2
        bt.Sequence([
            bt.SpotCleaning(),
            bt.LowBatteryInterrupt(
            btl.Timer(20,bt.CleanSpot()) ), 
            bt.DoneSpot()
            ]),


        bt.Sequence([
            bt.LowBatteryInterrupt(
            bt.GeneralCleaning()),  # Only runs if this condition is true

            bt.Sequence([
                bt.Selection([  # Priority: 1 = DustySpot; 2 = CleanFloor
                    bt.Sequence([
                        bt.DustySpot(),
                        bt.LowBatteryInterrupt(
                        btl.Timer(35, bt.CleanSpot()) ),
                        bt.AlwaysFail()  # Ensures fallback to next option
                            ]),
                    bt.LowBatteryInterrupt(
                    bt.UntilFails(bt.CleanFloor()))  # Priority node 2. Run CleanFloor until it fails 
                            ]),

                bt.DoneGeneral()  # Runs after selection is done
                    ]),
            
        ]),

        # Priority 3
        bt.Sequence([
            bt.DoNothing()
            ])
    ])
    return tree_root




# **** EXPERIMENTAL TREE ****
def experimental_version(current_blackbaord):
    tree_root = bt.Selection([
        # Priority 1 (Battery Check)
        bt.Sequence([bt.BatteryLessThan30(),
            bt.FindHome(),
            bt.GoHome(),
            bt.Docking(),
            bt.ChargingBattery() ]),
        
        # Priority 2 (AI: Pet Recognition)
          bt.Sequence([
            bt.LowBatteryInterrupt(
            bt.PetScanRecognition()),  # Only runs if this condition is true
            bt.Sequence([
                bt.Selection([  # Priority: [1 = Checking pet location; 2 = Continue other tasks]
                    bt.Sequence([
                        bt.CheckPetLocation(),
                        bt.LowBatteryInterrupt(
                        btl.Timer(10, bt.CleanSpot()) ),
                        bt.DoneCleanFloor()  # Ensures fallback to next option
                            ]),
                    bt.LowBatteryInterrupt(
                    bt.DoOtherTasks())  
                            ]),  # Runs after selection is done
                    ]),            
        ]),
        
        
        
        # Priority 3 (Spot cleaning and Genral cleaning)
        bt.Sequence([
            bt.SpotCleaning(),
            bt.LowBatteryInterrupt(
            btl.Timer(20,bt.CleanSpot()) ), 
            bt.DoneSpot()
            ]),


        bt.Sequence([
            bt.LowBatteryInterrupt(
            bt.GeneralCleaning()),  # Only runs if this condition is true

            bt.Sequence([
                bt.Selection([  # Priority: 1 = DustySpot; 2 = CleanFloor
                    bt.Sequence([
                        bt.DustySpot(),
                        bt.LowBatteryInterrupt(
                        btl.Timer(35, bt.CleanSpot()) ),
                        bt.AlwaysFail()  # Ensures fallback to next option
                            ]),
                    bt.LowBatteryInterrupt(
                    bt.UntilFails(bt.CleanFloor()))  # Priority node 2. Run CleanFloor until it fails 
                            ]),

                bt.DoneGeneral()  # Runs after selection is done
                    ]),
            
        ]),

        # Priority 3  (Do Nothing)
        bt.Sequence([
            bt.DoNothing()
            ])
    ])
    return tree_root














    return 