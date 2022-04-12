'''

    3D toy dataset generator - adapted with reference to Alex Marshall's generator.py, see https://github.com/alexmarshallbristol/GNN_example

'''
from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import truncnorm
import pandas as pd #debugging
import sys #debugging

from time import perf_counter



def generate_3D(nTracks=10, plot = False, GaussBlur = 0, Cheat = False, Dense = False, scanning_radius = 0.01, undirectedEdges=True, **kwargs):

    if nTracks < 10:
        print("Minimum number of tracks is 10.")


    nTracks_Bmeson = np.random.randint(3,6) # Number of decay tracks originating from the B meson decay.

    nTracks_primaryVertex = nTracks - nTracks_Bmeson


    if plot == True:

        print("B meson tracks: {} \n Total tracks: {}".format(nTracks_Bmeson, nTracks))

    
    #generating parameters for the displaced vertex
    
    displaced_eta = np.random.uniform(low=2., high = 5.0, size = 1) #Generating pseudorapidity numbers with LHCb acceptance range for the B-meson decay track
    displaced_eta = displaced_eta * np.random.choice([1,-1], size = 1)

    displaced_theta = 2. * np.arctan(np.exp(-displaced_eta)) #angle relative to beamline
    displaced_phi = pi * np.random.uniform(low = 0., high = 2.0, size = 1) #angle around beam

    x_displacement = truncnorm.rvs(-1, 3, size=1)[0]*1.5+2. # Max displacement is 6.5
    
    r_from_beamline = x_displacement * np.tan(displaced_theta)[0]
 
    y_displacement = r_from_beamline * np.cos(displaced_phi)[0]
    z_displacement = r_from_beamline * np.sin(displaced_phi)[0]

    displaced_vertex = [x_displacement, y_displacement, z_displacement]

    #plotting parameters

    
    detector_planes = [7., 9., 11., 13.] #Detector planes at x = 7, 9 defined in a 3D detector space

    n_detectors = len(detector_planes)

    plot_x = detector_planes[-1] + 2. # x value to which particle tracks will be drawn

    array_length = n_detectors + 1 # Generalising to an arbitrary number of detector planes. +1 accounts for a flag for each track

    origin = np.empty((0,3)) #array to store the origin of each track (?)

    points_at_detector_planes = np.empty((array_length,3,0))

    

    #generate track parameters for all tracks

    etas = np.random.uniform(low=2., high = 5.0, size = nTracks)
    etas = etas * np.random.choice([1,-1], size = 1)

    thetas = 2. * np.arctan(np.exp(-etas)) #angles relative to the beamline


    phis = pi * np.random.uniform(low = 0., high = 2.0, size = nTracks) #circular distribution over 2 pi, describes how 'rotated' the angled track is

    x = np.empty(nTracks)
    y = np.empty(nTracks)
    z = np.empty(nTracks)

    for i in range(nTracks):

        plane_values = np.empty((0,3))

        if i < nTracks_primaryVertex:

            #generating tracks from the primary vertex

            x[i] = plot_x
            y[i] = x[i] * np.tan(thetas[i]) * np.cos(phis[i])
            z[i] = x[i] * np.tan(thetas[i]) * np.sin(phis[i])


            #Why is this formatted like this?

            if i == 0: #super hacky fix to an irritating bug

                points = [[0.,0.,0.], [x[i], y[i], z[i]], [0.,0.,0.]] #array to store the origin, final values, and a flag if originating from primary vertex or displaced vertex

            else: 

                points = np.dstack((points, [[0.,0.,0.], 
                                            [x[i], y[i], z[i]], 
                                            [0.,0.,0.]] 
                                ))

            #access values in points via [<ORIGIN>/<FINAL>/<FLAG>][Number of Dimensions][Number of Tracks]

            #eg. Final coordinate values of track 14 = [1,:,13], returning [x,y,z]

                        
            
                                
            for j in range(array_length-1): # [x,y,z] values of track i at plane j

                plane_val = [
                            detector_planes[j], 
                            detector_planes[j] * np.tan(thetas[i]) * np.cos(phis[i]), 
                            detector_planes[j] * np.tan(thetas[i]) * np.sin(phis[i]) 
                            ]
                
                plane_values = np.vstack((plane_values, plane_val))

            plane_values =  np.vstack((plane_values, [0.,0.,0.])) #adding a flag to say that the track originated from the primary vertex

            origin = np.append(origin, [[0.,0.,0.]], axis = 0)


            points_at_detector_planes = np.dstack((points_at_detector_planes, plane_values)) # stacks the values on all the planes into a 3D array of final dimensions:
                                                                                             #[num_detectors][3 (number of dimensions)][num_tracks]

            # to access values from nth track, take array elements [:,:,n]
            # to access values from the nth track at the qth detector [q,:,n][0/1/2] where 0/1/2 = x/y/z
            # the final value from each track will be the flag ie. [-1,:,n]

        else:

            #generating tracks from the displaced vertex - angular distribution may be fucked here... or may be fixed if momentum conservation is added as a constraint...

            x[i] = plot_x - displaced_vertex[0]
            y[i] = x[i] * np.tan(thetas[i] + displaced_theta) * np.cos(phis[i])      # + displaced_phi)
            z[i] = x[i] * np.tan(thetas[i] + displaced_theta) * np.sin(phis[i])      # + displaced_phi)

            points = np.dstack((points, [[displaced_vertex[0],displaced_vertex[1],displaced_vertex[2]],
                                         [x[i] + displaced_vertex[0], y[i] + displaced_vertex[1], z[i] + displaced_vertex[2]],
                                         [1.,1.,1.]],                                       
                               
                               ))

            for j in range(array_length-1): # [x,y,z] values of track i at plane j - displaced this time

                plane_val = [
                             detector_planes[j], 
                            ((detector_planes[j]- displaced_vertex[0]) * np.tan(thetas[i]+displaced_theta) * np.cos(phis[i])).item() + displaced_vertex[1],    #+displaced_phi), 
                            ((detector_planes[j]- displaced_vertex[0]) * np.tan(thetas[i]+displaced_theta) * np.sin(phis[i])).item() + displaced_vertex[2]   #+displaced_phi) 
                            ]
                
                plane_values = np.vstack((plane_values, plane_val))

            plane_values =  np.vstack((plane_values, [1.,1.,1.]))  #adding a flag to say that the track originated from the primary vertex

            

            points_at_detector_planes = np.dstack((points_at_detector_planes, plane_values))

            origin = np.append(origin, [displaced_vertex], axis = 0)

    #reshaping

    labels = points_at_detector_planes[-1,:,:][0]
    input = points_at_detector_planes[0:-1,:,:]

    origins = origin[:,:, np.newaxis] 
    origins = np.repeat(origins[:,:], repeats = n_detectors, axis = 2) 

    origins = np.moveaxis(origins, 1, 2)
    origins = np.moveaxis(origins, 1, 0)
 
    input = np.moveaxis(input, 1, 2) #swapping xyz onto third axis for easier debugging
    labels = np.repeat(labels[:,np.newaxis], n_detectors,axis = 1)



    #Generating a gaussian noise array and adding it to the y, z coordinates on the detector
    
    if GaussBlur != 0:
        
        noise = np.random.normal(0, GaussBlur, size = (n_detectors,nTracks, 2))

        input[:,:,1:] += noise
    
    input = input.reshape(-1, input.shape[-1]) #reshaping into a list of [n_nodes, node_coords] to be compatible with graphs
                                               #n_nodes = n_tracks x n_detectors
    labels = labels.flatten(order = 'F')


    #Generating the adjacency matrix - code splits into two here

    if Cheat == False:

        adjacency_matrix = MakeAdjancency(nTracks, n_detectors, Cheat, Dense)

        if Dense == False:
            for i in range(n_detectors-2): #scan edges between planes

                edge_list = np.moveaxis(np.where(adjacency_matrix[i*nTracks:(i+1)*nTracks]), 0,1)

                edge_mask = np.full(edge_list.shape, False)

                edge_list[:,0] += i*nTracks #fixing slice index

                #Generate edge features for non-zero edges
                
                diffs = input[edge_list[:,1],:] - input[edge_list[:,0],:]

                diff_phi = np.arctan2(diffs[:,2], diffs[:,1])
                diff_theta = np.arctan(np.sqrt(diffs[:,1]**2 + diffs[:,2]**2) / 2.)

                edge_features = np.stack((diff_theta, diff_phi), axis = 1)

                predicted_vals = np.moveaxis(np.array([input[edge_list[:,1], 0]+2, 
                                    input[edge_list[:,1], 1]+(2*np.tan(edge_features[:,0])*np.cos(edge_features[:,1])),
                                    input[edge_list[:,1], 2]+(2*np.tan(edge_features[:,0])*np.sin(edge_features[:,1]))]
                                    ), 0,1)

                for j in range(edge_list.shape[0]):

                    test = np.moveaxis(np.where(np.sqrt(np.sum((input[(i+2)*nTracks:(i+3)*nTracks] - predicted_vals[j])**2, axis = 1)) < scanning_radius), 0,1)

                    #print(test)
                    #print(test.size)


                    if test.size == 0:

                        #print("Deleting edge [{},{}]".format(edge_list[j,0],edge_list[j,1]))
                        
                        adjacency_matrix[edge_list[j,0],edge_list[j,1]] = 0

                    else:
                        
                        #print("Creating edge between [{},{}]".format(edge_list[j,1],(i+2)*nTracks+test[:]))

                        adjacency_matrix[edge_list[j,1],(i+2)*nTracks+test[:]] = 1

                        #edge_mask[j] = [False,False]

    elif Cheat == True:

        adjacency_matrix = MakeAdjancency(nTracks, n_detectors, Cheat, False)

    #input takes the form [ndetectors, ntracks , x/y/z] -> input[0,0,:] is the coordinates of the point at detector 0 from track 0


    if undirectedEdges == True:

        adjacency_matrix = adjacency_matrix +adjacency_matrix.T

    #Grab edge features of the finalised edges:

    edge_list = np.moveaxis(np.where(adjacency_matrix), 0,1)
    diffs = input[edge_list[:,1],:] - input[edge_list[:,0],:]

    diff_phi = np.arctan2(diffs[:,2], diffs[:,1])
    diff_theta = np.arctan(np.sqrt(diffs[:,1]**2 + diffs[:,2]**2) / 2.)

    edge_features = np.stack((diff_theta, diff_phi), axis = 1)

    if plot == True:

        print("B meson tracks: {} \n Total tracks: {}".format(nTracks_Bmeson, nTracks))

        fig = plt.figure()

        ax = plt.axes(projection ='3d')

        #plot the particle tracks

        for i in range(nTracks):

            if i < nTracks_primaryVertex:
                plt.plot(points[:,0,i][0:2],points[:,1,i][0:2],zs = points[:,2,i][0:2], color='tab:blue',alpha=0.25)
            else:
                plt.plot(points[:,0,i][0:2],points[:,1,i][0:2],zs = points[:,2,i][0:2], color='tab:red')

        ax.scatter(input[:,0], input[:,1], input[:,2], marker = 'x', alpha=0.5)

        red_patch = mpatches.Patch(color='crimson', label='Displaced Vertex Tracks')
        blue_patch = mpatches.Patch(color='skyblue', label='Primary Vertex Tracks')

        plt.legend(handles=[red_patch, blue_patch])

        #plot the planes

        y_lim = np.array(ax.get_ybound())
        z_lim = np.array(ax.get_zbound())

        ys = np.linspace(y_lim[0], y_lim[1], 25)
        zs = np.linspace(z_lim[0], z_lim[1], 25)


        (yy, zz) = np.meshgrid(y_lim, z_lim)
        
        for i in range(n_detectors):

            ax.plot_surface(detector_planes[i] ,yy, zz, alpha = 0.2)

        plt.show()


    return input, origins, labels, adjacency_matrix, edge_features, n_detectors, nTracks


def MakeAdjancency(submatrix_dim, large_matrix_dim, Cheat=False, Dense=True, **kwargs): #construct adjacency matrix from smaller submatrices

    if Cheat == False: #Fully connected on adjacent planes

        if Dense == True:

            A = np.ones((submatrix_dim, submatrix_dim))
            B = np.zeros((submatrix_dim, submatrix_dim))

            matrix_structure = np.diag(np.ones(large_matrix_dim-1), 1)

        else:

            A = np.ones((submatrix_dim, submatrix_dim))
            B = np.zeros((submatrix_dim, submatrix_dim))

            initialiser = np.zeros(large_matrix_dim-1)
            initialiser[0] = 1

            matrix_structure = np.diag(initialiser, k = 1)
            # + np.diag(np.ones(large_matrix_dim-1), -1)

    elif Cheat == True: #Passing track information into adjacency matrix

        A = np.identity(submatrix_dim)
        B = np.zeros((submatrix_dim, submatrix_dim))

        matrix_structure = np.triu(np.ones((large_matrix_dim,large_matrix_dim)), k =1)

    ms = str(np.where(matrix_structure.astype(bool), "A", "B").tolist()).replace("'A'", "A").replace("'B'", "B")

    return np.block(eval(ms))

