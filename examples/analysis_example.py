'''

Example of a small script to perform analysis on a generic gsd_file


'''


from SAASH import analyze
from SAASH.util import observer as obs
import sys


def setup_observer(gsd_file, run_type, observables = None, jump=1):

    #init an observables with the file and run type
    observer = obs.Observer(gsd_file, run_type, jump = jump)

    #for cluster runs, set up all the desired observables
    if run_type == 'cluster' and observables is not None:
        for observable in observables:
            observer.add_observable(observable)

    #change the start and end frames
    # observer.set_first_frame(100)
    # observer.set_final_frame(400)

    return observer


if __name__ == "__main__":

    #get command line args for gsd_file, ixn_file, and number of frames to jump
    try:
        gsd_file = sys.argv[1]
        ixn_file = sys.argv[2]
        jump     = int(sys.argv[3])
    except:
        print("Usage: %s <gsd_file> <ixn_file> <frame_skip>" % sys.argv[0])
        raise

    #do a bulk analysis run - tracks number of clusters of each size every jump frames
    observer = setup_observer(gsd_file, 'bulk', jump=jump)
    analyze.run_analysis(gsd_file, ixn_file=ixn_file, observer=observer)

    #do a cluster analysis with num_bodies as an observable
    observables = ['num_bodies']
    observer = setup_observer(gsd_file, 'cluster', observables=observables)
    analyze.run_analysis(gsd_file, ixn_file=ixn_file, observer=observer)