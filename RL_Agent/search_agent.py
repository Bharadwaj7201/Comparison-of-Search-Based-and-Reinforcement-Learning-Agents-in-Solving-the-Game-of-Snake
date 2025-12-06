# search_agent.py (SAFE, RULE-BASED AGENT)

import numpy as np

class SearchAgent:
    def __init__(self, game):
        self.game = game

    def get_action(self, state):
        """
        Simple, stable rule-based agent using the 11-feature state.
        Never produces invalid directions.
        """

        danger_straight = state[0]
        danger_right = state[1]
        danger_left = state[2]

        food_left  = state[3]
        food_right = state[4]
        food_up    = state[5]
        food_down  = state[6]

        dir_l = state[7]
        dir_r = state[8]
        dir_u = state[9]
        dir_d = state[10]

        # ------------------------------------------------------
        # 1. Avoid danger first
        # ------------------------------------------------------
        if danger_straight:
            if not danger_left:
                return 2   # turn left
            if not danger_right:
                return 1   # turn right
            return 0       # forced straight

        # ------------------------------------------------------
        # 2. Move toward food safely
        # ------------------------------------------------------
        if dir_l or dir_r:     # moving horizontally
            if food_up and not danger_left:
                return 2
            if food_down and not danger_right:
                return 1

        if dir_u or dir_d:     # moving vertically
            if food_left and not danger_left:
                return 2
            if food_right and not danger_right:
                return 1

        # ------------------------------------------------------
        # 3. Default behavior
        # ------------------------------------------------------
        return 0  # straight
