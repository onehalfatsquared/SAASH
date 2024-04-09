# Self Assembly Analysis Suite for HOOMD (SAASH)

This repository contains a general analysis script for tracking the progress of self-assembling subunits simulated using HOOMD. 

The user provides an input interaction file that lists all the pairs of pseudoatoms that are responsible for bonds in addition to the cutoff distance for that interaction. User can optionally provide a pseudoatom label for any nanoparticles in the domain and their radii, enabling the tracking of the largest cluster assembled around the nanoparticle. 

The analysis currently has 3 modes; bulk, nanoparticle, and cluster. 
Bulk will report the total number of clusters of each size on each frame analyzed. 
Nanoparticle only tracks assembly that occurs in the vicinity of a nanoparticle scaffold. For each frame analyzed, reports the number of subunits attached to the nanoparticle, the size of the largest connected component of those subunits, and the total number of bonds between the subunits. 
Cluster constructs a time series of user specified properties of a cluster (e.g. number of subunits, bond distribution, positions, subunit ids, etc). If cluster's merge or split, they are assigned parents or children. 

This library was developed for systems with subunits that bind via edge-mediated interactions. It should work as long as subunits can be considered bonded if a pair of pseudoatoms is within a cutoff distance, i.e. isotropic attractive spheres or geometric objects with attractive patches on edges. There may be workarounds for other cases. For example, pentagonal subunits with attractors at the vertices can be analyzed by adding inert tracer particles to the edge midpoints. 


## Package Install Instructions
### Local Machine

If you are installing to a local machine, first clone into the repo. From the highest SAASH directory (that contains setup.py), run the following commands:


`python setup.py build`

`python setup.py install`




### Cluster Install

If the above approach does not work, or you are installing to somewhere where you do not have sudo priveleges (like a compute cluster), an alternative approach has been to do the following:

`python setup.py bdist_wheel`

`pip install dist/SAASH-0.1.0-py3-none-any.whl --force-reinstall --user`

I think this approach requires the wheel and twine packages. The exact filename inside the dist folder will change with version, so just look for the .whl file created when running the setup script.
