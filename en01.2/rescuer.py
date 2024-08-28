##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position


import os
import random
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map: Map = None             # explorer will pass the map
        self.victims = None         # list of found victims
        self.plan = []              # a list of planned actions
        self.has_victims = set()   # positions already planned to be visited
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan

                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    
    def go_save_victims(self, map, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        print(f"\n\n*** R E S C U E R ***")
        self.map = map
        print(f"{self.NAME} Map received from the explorer")
        self.map.draw()

        print()
        #print(f"{self.NAME} List of found victims received from the explorer")
        self.victims = victims

        # print the found victims - you may comment out
        #for seq, data in self.victims.items():
        #    coord, vital_signals = data
        #    x, y = coord
        #    print(f"{self.NAME} Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")

        #print(f"{self.NAME} time limit to rescue {self.plan_rtime}")

        self.__planner()
                  
        self.set_state(VS.ACTIVE)

    def __planner(self):
        """ A private method that calculates the walk actions in a OFF-LINE MANNER to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        """ This plan starts at origin (0,0) and chooses the first of the possible actions in a clockwise manner starting at 12h.
        Then, if the next position was visited by the explorer, the rescuer goes to there. Otherwise, it picks the following possible action.
        For each planned action, the agent calculates the time will be consumed. When time to come back to the base arrives,
        it reverses the plan."""

        last_pos = self.map.get((0, 0))

        self.plan = []
        for v in self.victims:
            self.plan += self.map.get_path(last_pos, v["coords"], self)[1:]
            last_pos = v["coords"]
            self.has_victims.add(v["coords"])

        come_back_plan = self.map.get_path(last_pos, self.map.get((0, 0)), self)

        self.plan = self.plan + come_back_plan

    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           #input(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        pos = self.plan.pop(0)

        dx = self.x + pos.coords[0]
        dy = self.y + pos.coords[1]
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # check if there is a victim at the current position
            if pos in self.has_victims:
                rescued = self.first_aid() # True when rescued
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        #input(f"{self.NAME} remaining time: {self.get_rtime()} Tecle enter")

        return True

