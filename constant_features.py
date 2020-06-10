import numpy as np
import pandas as pd


def remove_constants(state):
    if type(state) == pd.DataFrame:
        return state.drop(columns=[str(i) for i in constant_features.keys()])
    else:  # List / array
        return np.delete(state, list(constant_features.keys()))


def add_constants(state):
    if type(state) == pd.DataFrame:
        for k, v in constant_features.items():
            state[k] = v
        state = state[[i for i in range(310)]]
        return state
    else:  # List / array
        for k, v in constant_features.items():
            state = np.insert(state, k, v)
        return state


constant_features = {
    0: 2.000,  # team
    8: 305.000,  # base move speed
    9: 0.000,  # current move speed
    11: 9.000,  # damage variance
    13: 250.000,  # attack range buffer
    16: 0.500,  # attack animation point
    25: 0.000,  # denies
    30: 3.000,  # opp team
    38: 305.000,  # opp base move speed
    39: 0.000,  # opp curr move speed
    41: 9.000,  # opp damage variance
    43: 250.000,  # opp attack range buffer
    46: 0.500,  # opp attack animation point
    57: 2.000,  # good tower team
    59: 1800.000,  # good tower max health
    61: 250.000,  # good tower attack range buffer
    62: 1.000,  # good tower attack speed
    63: 3.000,  # bad tower team
    65: 1800.000,  # bad tower max health
    67: 250.000,  # bad tower attack range buffer
    68: 1.000,  # bad tower attack speed
    69: -69.000,  # bounty 1 location 1
    70: -70.000,  # bounty 1 location 2
    71: -71.000,  # bounty 2 location 1
    72: -72.000,  # bounty 2 location 2
    73: -73.000,  # bounty 3 location 1
    74: -74.000,  # bounty 3 location 2
    75: -75.000,  # bounty 4 location 1
    76: -76.000,  # bounty 4 location 2
    77: -77.000,  # power 1 location 1
    78: -78.000,  # power 1 location 2
    79: -79.000,  # power 2 location 1
    80: -80.000,  # power 2 location 2
    226: 90.000,  # ability 1 mana cost
    227: 0.000,  # ability 1 ability damage
    228: 200.000,  # ability 1 cast range
    230: 0.000,  # ability 1 target type
    231: 0.000,  # ability 1 behavior
    233: 90.000,  # ability 2 mana cost
    234: 0.000,  # ability 2 ability damage
    235: 450.000,  # ability 2 cast range
    237: 0.000,  # ability 2 target type
    238: 0.000,  # ability 2 behavior
    240: 90.000,  # ability 3 mana cost
    241: 0.000,  # ability 3 ability damage
    242: 700.000,  # ability 3 cast range
    244: 0.000,  # ability 3 target type
    245: 0.000,  # ability 3 behavior
    247: 0.000,  # ability 4 mana cost
    248: 0.000,  # ability 4 ability damage
    249: 0.000,  # ability 4 cast range
    250: 0.000,  # ability 4 cooldown time remaining
    251: 0.000,  # ability 4 target type
    252: 0.000,  # ability 4 behavior
    253: 0.000,  # ability 5 level
    254: 0.000,  # ability 5 mana cost
    255: 0.000,  # ability 5 ability damage
    256: 0.000,  # ability 5 cast range
    257: 0.000,  # ability 5 cooldown time remaining
    258: 0.000,  # ability 5 target type
    259: 0.000,  # ability 5 behavior
    260: 0.000,  # ability 6 level
    261: 150.000,  # ability 6 mana cost
    262: 0.000,  # ability 6 ability damage
    263: 0.000,  # ability 6 cast range
    264: 0.000,  # ability 6 cooldown time remaining
    265: 0.000,  # ability 6 target type
    266: 0.000,  # ability 6 behavior
    268: 90.000,  # opp ability 1 mana cost
    269: 0.000,  # opp ability 1 ability damage
    272: 0.000,  # opp ability 1 target type
    273: 0.000,  # opp ability 1 behavior
    275: 90.000,  # opp ability 2 mana cost
    276: 0.000,  # opp ability 2 ability damage
    279: 0.000,  # opp ability 2 target type
    280: 0.000,  # opp ability 2 behavior
    282: 90.000,  # opp ability 3 mana cost
    283: 0.000,  # opp ability 3 ability damage
    286: 0.000,  # opp ability 3 target type
    287: 0.000,  # opp ability 3 behavior
    289: 0.000,  # opp ability 4 mana cost
    290: 0.000,  # opp ability 4 ability damage
    291: 0.000,  # opp ability 4 cast range
    292: 0.000,  # opp ability 4 cooldown time remaining
    293: 0.000,  # opp ability 4 target type
    294: 0.000,  # opp ability 4 behavior
    295: 0.000,  # opp ability 5 level
    296: 0.000,  # opp ability 5 mana cost
    297: 0.000,  # opp ability 5 ability damage
    298: 0.000,  # opp ability 5 cast range
    299: 0.000,  # opp ability 5 cooldown time remaining
    300: 0.000,  # opp ability 5 target type
    301: 0.000,  # opp ability 5 behavior
    302: 0.000,  # opp ability 6 level
    303: 150.000,  # opp ability 6 mana cost
    304: 0.000,  # opp ability 6 ability damage
    305: 0.000,  # opp ability 6 cast range
    306: 0.000,  # opp ability 6 cooldown time remaining
    307: 0.000,  # opp ability 6 target type
    308: 0.000,  # opp ability 6 behavior
}
