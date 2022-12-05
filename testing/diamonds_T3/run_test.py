from SAASH import analyze
from SAASH.util import observer as obs

#import profiling tools
import cProfile, pstats
import re
import time


def setup_observer(gsd_file, jump = 1):

    run_type = 'nanoparticle'
    observer = obs.Observer(gsd_file, run_type, jump=jump)

    observer.set_first_frame(400)
    observer.set_final_frame(900)

    return observer


def run_profile():

    # test_file = "T3_triangles.gsd"
    test_file = "sd1296.gsd"
    ixn_file  = "diamond_ixn.txt"

    observer = setup_observer(test_file, jump = 25)
    analyze.run_analysis(test_file, ixn_file=ixn_file, observer=observer)



if __name__ == "__main__":

    #test run
    # run_profile()

    #profiling for speed
    cProfile.run('run_profile()', 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs().sort_stats('tottime').print_stats(20)

