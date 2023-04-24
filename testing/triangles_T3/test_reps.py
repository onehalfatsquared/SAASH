from __future__ import print_function
import sys, os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import gsd, gsd.pygsd, gsd.hoomd

import numpy as np
from SAASH.util.state import StateRep
from SAASH.util.state import StateRepCollection


#CRITICAL: If things are pickled within the util module of SAASH, 
#the following is required to unpickle outside of that project's namespace
from SAASH import util
sys.modules['util'] = util


def srep_playground():

    database_loc = "T3_triangles.srep"
    srep_collection = StateRepCollection(database_loc)

    state_dict = srep_collection.get_dict()
    state_keys = list(state_dict.keys())

    ex_state = state_keys[10]
    ex_rep   = srep_collection.get_rep(ex_state)

    print(ex_rep.get_size())
    print(ex_rep.get_positions())
    print(ex_rep.get_types())
    print(ex_rep.get_orientations())

    sys.exit()

    input_file = "../simulation/input.conf"
    inputs     = InputManager(input_file)

    TargetSnap(inputs, )
    

    return

if __name__ == "__main__":

    srep_playground()