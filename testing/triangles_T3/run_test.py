'''

Example of a small script to perform analysis on a generic gsd_file


'''


from SAASH import analyze
from SAASH.util import observer as obs
import sys


def setup_observer(gsd_file, run_type, observables = None, jump=1):

    #init an observables with the file and run type
    observer = obs.Observer(gsd_file, run_type, jump = jump)
    observer.set_focus_list([2,10,60])

    #for cluster runs, set up all the desired observables
    if observables is not None:
        for observable in observables:
            observer.add_observable(observable)

    #change the start and end frames
    observer.set_first_frame(0)
    # observer.set_final_frame(3000)

    return observer


if __name__ == "__main__":

    test_file = "T3_triangles.gsd"
    # test_file = "nano_test.gsd"
    ixn_file  = "interactionsT3.txt"

    jump = 100
    
    observables = ['num_bodies', 'indices', 'bonds', 'bond_counts']

    #do a bulk analysis run - tracks number of clusters of each size every jump frames
    observer = setup_observer(test_file, 'bulk', jump=jump, observables=observables)
    analyze.run_analysis(test_file, ixn_file=ixn_file, observer=observer)

    #do a cluster analysis with num_bodies as an observable
    observer = setup_observer(test_file, 'cluster', observables=observables, jump=jump)
    analyze.run_analysis(test_file, ixn_file=ixn_file, observer=observer)


