'''

This file implements a neighborgrid data structure, which allows for more efficient
sorting of particles by their positions. By only searching nearby grid cells, the 
number of pairwise distance computations to determine bonds can be drastically reduced. 

Code modified from a 2D version created by Daniel Goldstein. 

'''


import numpy as np
from itertools import permutations
import sys


class Neighborgrid:
    '''This class implements a neighborlist to faciliate the fast
    finding of nearby partilces. This implementation is for a 2D
    neighbor grid'''

    def __init__(self, lim, R, periodic):
        
        #set the bounding box for the simulation
        self.lim = lim

        #set which dimensions are periodic
        self.periodic = periodic

        #set the interaction range
        self.R = R

        #get the number of dimensions for the requested system
        self.dim = len(self.lim)
        
        #break up the space of the simulation into a grid were the size
        #of each box is at least R/2 then we will only need to check the
        #9 boxes around each particle

        self.numD    = np.zeros(self.dim, dtype=int)
        self.boxSize = np.zeros(self.dim, dtype=float)

        for i in range(self.dim):
            self.numD[i]    = np.floor((self.lim[i][1] - self.lim[i][0]) / (R/2))
            self.boxSize[i] = (self.lim[i][1] - self.lim[i][0]) / self.numD[i]
        
        self.map = {}

    def update(self, bodies):
        '''The update function takes in a list of bodies (objects with a position) and will
        create a map from box number and from particle'''

        #empty out the dictionary from previous step
        self.map.clear()

        #add in new particle positions
        for body in bodies:
            self.addItemToMap(self.convertPosToIndex(body), body)
            


    def convertPosToIndex(self, body):

        #grab the position of the body (2 or 3 dim)
        position = body.get_position()

        #check if all coordinates are within the known box size
        for i in range(self.dim):

            if position[i] < self.lim[i][0] or position[i] > self.lim[i][1]:
                raise ValueError("Particle is outside the set bounds")

        #convert to an index - tuple for use as key in map
        index = []
        for i in range(self.dim):

            box_num = np.floor((position[i]+self.lim[i][1]) / self.boxSize[i])
            index.append(int(box_num))

        # print(position, index)
        # sys.exit()
        return tuple(index)



    def addItemToMap(self, key, value):

        if key in self.map:
            self.map[key].append(value)

        else:
            self.map[key] = [value]


    def getNeighborhood(self, body):
        '''find the box that the atom is in, look in the 5x5(x5) area around
        it and return all that are within the interaction radius'''

        centerBox = self.convertPosToIndex(body)  
        body_list = []

        for box in self.AdjacentBoxes(centerBox):
            if box in self.map:
                if isinstance(self.map[box], list):
                   for body_j in self.map[box]:
                        '''was previously a distance check here, but interactions are not
                           at the body level, so I removed it for purposes of gridding'''
                        yield body_j
               
       
    def AdjacentBoxes(self, centerBox):
        ''' determine all adjacent boxes to the center'''

        #get all possible grid shifts within range
        indexAdjustment = permutations([-2,-1,0,1,2], self.dim)
        print("All adjust")
        print(list(indexAdjustment))

        # add the adjustment to the center box and then wrap boundaries  
        print("Center ", centerBox)
        print("Neighbors")
        for boxAdjustment in indexAdjustment:
            neighborBox = [0]*self.dim
            for i in range(self.dim):
                neighborBox[i] = (centerBox[i]+boxAdjustment[i]) % self.numD[i]
            print(neighborBox)

            yield tuple(neighborBox)


