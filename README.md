# Self Assembly Analysis Script for HOOMD (SAASH)

This repository contains a general analysis script for tracking the progress of self-assembling subunits simulated using HOOMD. 

The user provides an input interaction file that lists all the pairs of pseudoatoms that are responsible for bonds in addition to the cutoff distance for that interaction. User can optionally provide a pseudoatom label for any nanoparticles in the domain and their radii, enabling the tracking of the largest cluster assembled around the nanoparticle. 

To be added to later. 


#Package Install Instructions (For Now)
To be able to import this code as a package outside of the source directory, I have been using a python setup file to make a pip package and install it. It is not published yet (maybe I will do this eventually, after finalizing and documenting some things), but this works for now. 

If you are installing to a local machine, first clone into the repo. From the highest SAASH directory (that contains setup.py), run the following commands:

python setup.py build

python setup.py install


This should allow SAASH to be imported from anywhere. 

If the above approach does not work, or you are installing to somewhere where you do not have sudo priveleges (like a compute cluster), an alternative approach has been to do the following:

python setup.py bdist_wheel

pip install dist/SAASH-0.1.0-py3-none-any.whl --force-reinstall --user

I think this approach requires the wheel and twine packages. The exact filename inside the dist folder may change, so just look for the .whl file when running the command. 
