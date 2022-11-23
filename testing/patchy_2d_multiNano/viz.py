import gsd.hoomd
import numpy as np
import PIL
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import subprocess

import os


from matplotlib.animation import FuncAnimation


def plotFrame(gsd_file, frame):
    #use matplotlib to plot the structure

    '''
    Types map:
    0 <-> C
    1 <-> S
    2 <-> P
    '''

    #open the file and get the requested frame
    with gsd.hoomd.open(name=gsd_file, mode="rb") as gsd_file:
        snap = gsd_file[frame]

    #get the box data
    box = snap.configuration.box
    L = [box[0], box[1], box[2]]

    #get total number of particles
    Ntot = snap.particles.N

    #get particle types map
    particle_types = snap.particles.typeid

    #get the positions and number of particles of type 0
    centers = snap.particles.position[particle_types == 0]
    N = len(centers)

    #get the scaffold sphere position
    sp = snap.particles.position[particle_types == 1]
    # print(centers, sp)
    print(sp)

    #get the patch locations
    patches = snap.particles.position[particle_types == 2]

    #get the repelers
    repels  = snap.particles.position[particle_types == 3]

    #get the repelers
    attracts  = snap.particles.position[particle_types == 4]

    #make a figure to add circles
    fig, ax = plt.subplots()
    ax.set_xlim((-L[0]/2-1, L[0]/2+1))
    ax.set_ylim((-L[1]/2-1, L[1]/2+1))

    #loop over centers and add circles of radius 0.5
    for i in range(N):
        center = (centers[i][0], centers[i][1])
        circle = plt.Circle(center, 0.5, color='blue')
        ax.add_patch(circle)

    #add the scaffold
    for i in range(len(sp)):
        scenter = (sp[i][0], sp[i][1])
        circle = plt.Circle(scenter, 0.5, color='red')
        ax.add_patch(circle)

    #add the patches
    for i in range(2*N):
        center = (patches[i][0], patches[i][1])
        circle = plt.Circle(center, 0.04, color='yellow')
        ax.add_patch(circle)

    #add repelers
    for i in range(N):
        center = (repels[i][0], repels[i][1])
        circle = plt.Circle(center, 0.04, color='orange')
        ax.add_patch(circle)

    #add attracts
    #for i in range(N):
     #   center = (attracts[i][0], attracts[i][1])
      #  circle = plt.Circle(center, 0.04, color='green')
       # ax.add_patch(circle)

    plt.show()


def makeMovie():

    #get the final snapshot
    filename = "../trajectories/E8.5S8.5/traj9.gsd"
    seed = filename.split('traj')[-1].split('.')[0]
    frameex = -1
    with gsd.hoomd.open(name=filename, mode="rb") as gsd_file:
        snap = gsd_file[frameex]

    #get the box data
    box = snap.configuration.box
    L = [box[0], box[1], box[2]]
    shift = [x/2 for x in L]

    #get snaps and number of frames
    snaps = gsd.hoomd.open(name=filename, mode="rb")
    frames = len(snaps)

    def animate(frame):
        print("Generating frame {}".format(frame))
        #get the box data
        snap = snaps[frame]
        box = snap.configuration.box
        L = [box[0], box[1], box[2]]

        #get total number of particles
        Ntot = snap.particles.N

        #get particle types map
        particle_types = snap.particles.typeid

        #get the positions and number of particles of type 0
        centers = snap.particles.position[particle_types == 0]
        N = len(centers)

        #get the scaffold sphere position
        sp = snap.particles.position[particle_types == 1][0]
        # print(centers, sp)

        #get the patch locations
        patches = snap.particles.position[particle_types == 2]

        #get the repelers
        repels  = snap.particles.position[particle_types == 3]

        #get the repelers
        attracts  = snap.particles.position[particle_types == 4]

        #make a figure to add circles
        fig, ax = plt.subplots()
        ax.set_xlim((-7, 7))
        ax.set_ylim((-7, 7))

        #loop over centers and add circles of radius 0.5
        for i in range(N):
            center = (centers[i][0], centers[i][1])
            circle = plt.Circle(center, 0.5, color='blue')
            ax.add_patch(circle)

        #add the scaffold
        scenter = (sp[0], sp[1])
        circle = plt.Circle(scenter, 0.5, color='red')
        ax.add_patch(circle)

        #add the patches
        for i in range(2*N):
            center = (patches[i][0], patches[i][1])
            circle = plt.Circle(center, 0.04, color='yellow')
            ax.add_patch(circle)

        #add repelers
        for i in range(N):
            center = (repels[i][0], repels[i][1])
            circle = plt.Circle(center, 0.04, color='orange')
            ax.add_patch(circle)

        #add attracts
        for i in range(N):
            center = (attracts[i][0], attracts[i][1])
            circle = plt.Circle(center, 0.04, color='green')
            ax.add_patch(circle)

        #save result 
        if not os.path.exists('movie{}'.format(seed)):
            os.makedirs('movie{}'.format(seed))
        plt.savefig("movie{}/frame{}.png".format(seed, frame))
        plt.close()

    #get the frames
    for i in range(frames):
        if not os.path.exists('movie{}/frame{}.png'.format(seed,i)):
            animate(i)
        
    #load frames into array
    images = []
    for i in range(frames):
        img = Image.open("movie{}/frame{}.png".format(seed, i))
        images.append(img.copy())
        img.close()
    images[0].save('movie{}.gif'.format(seed), save_all=True, append_images=images[1:], duration=40, loop=0)





if __name__ == "__main__":
    
    gsd_file = "traj0.gsd"
    frame = 753
    plotFrame(gsd_file, frame)

    # makeMovie()

