from SAASH import analyze
from SAASH.util import observer as obs






def setup_observer(gsd_file):

    run_type = 'nanoparticle'
    observer = obs.Observer(gsd_file, run_type)

    return observer







if __name__ == "__main__":

    # test_file = "T3_triangles.gsd"
    test_file = "sd1296.gsd"
    ixn_file  = "diamond_ixn.txt"

    observer = setup_observer(test_file)

    analyze.run_analysis(test_file, ixn_file=ixn_file, jump=25, observer=observer)

