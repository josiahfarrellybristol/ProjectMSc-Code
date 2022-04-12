import os
import numpy as np
import scipy.sparse as sp
import time
from spektral.data import Dataset, Graph
from Generate3D_EdgeFull import generate_3D


class getData(Dataset):

    def __init__(self, n_samples = 1, plotting=False, n_min=10, n_max=25, newpath = None, Gauss = 0, Cheat=False, R_scan=0.01, Save = True, Dense = False, undirected = True, **kwargs):

        self.n_samples = n_samples
        self.n_min = n_min
        self.n_max = n_max
        self.plotting = plotting
        self.newpath  = os.path.join(newpath, f"NG_{n_samples}_TR{n_min}-{n_max}_Gauss{Gauss}_Scan{R_scan}_Cheatis_{Cheat}")
        self.gauss = Gauss
        self.Cheat = Cheat
        self.R_scan = R_scan
        self.Save = Save
        self.Dense = Dense
        self.undirected = undirected

        print(f"Undirected is : {self.undirected}")

        if Dense == True:

            self.newpath = f"{self.newpath}_Denseis_{Dense}"

        if undirected == False:

            self.newpath = f"{self.newpath}_DirectedEdges"

        super().__init__(**kwargs)

    
    # def path(self, input):
    #     self._path = os.path.join(input, f"NG_{self.n_samples}_TR{self.n_min}-{self.n_max}_Gauss{self.gauss}_Scan{self.R_scan}_Cheatis_{self.Cheat}")

    def download(self): #very hacky method of creating saved datasets - relies on ~spektral\datasets\getData not existing. 

        if (os.path.exists(self.newpath) == False)  and (self.Save == True):  #if the input path doesn't exist and initialiser is directed to save, creates a dataset in the path.
            os.mkdir(self.newpath) 
            print(f"Created directory {self.newpath}")

            for i in range(self.n_samples):
                num_tracks = np.random.randint(self.n_min, self.n_max)

                nodes_list, origins, node_labels, adjacency_list, edge_features, n_detectors, nTracks  = generate_3D(num_tracks, GaussBlur = self.gauss, 
                                                                                                                    Cheat=self.Cheat, scanning_radius=self.R_scan, 
                                                                                                                    undirectedEdges=self.undirected, Dense = self.Dense)

                filename = os.path.join(self.newpath, f'graph_{i}')
                np.savez(filename, x=nodes_list, a=adjacency_list, e=edge_features, y=node_labels)

    def read(self): #returns a list of n_samples of spektral.data.graph objects,


        output = []

        if os.path.exists(self.newpath) == True:  #reads saved graphs from the path if it exists
            directory = os.fsencode(self.newpath)

            print(f"Loading graphs from {self.newpath}")

            start = time.time()

            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                
                if filename.endswith(".npz"):
                    graph_data =  np.load(os.path.join(self.newpath, filename))
                    adjacency_list = sp.csr_matrix(graph_data['a']) #spektral expects sparse matrices
                    output.append(Graph(x=graph_data['x'], a=adjacency_list, e=graph_data['e'], y=graph_data['y']))
                else:
                    print("Non .npz file found in directory, skipping...")

            end = time.time()

            print(f"Loading took {end-start}s for {self.n_samples} graphs.")

        else: #if the path doesn't exist, generates new graphs without saving - mostly used for testing/validation
            start = time.time()
            print(f"No directory found, making {self.n_samples} new graphs")
            output = [self.make_graph() for _ in range(self.n_samples)]

            end = time.time()

            print(f"Generation took {end-start}s for {self.n_samples} graphs.")

        return output

    def make_graph(self):

        num_tracks = np.random.randint(self.n_min, self.n_max)

        nodes_list, origins, node_labels, adjacency_list, edge_features, n_detectors, nTracks  = generate_3D(num_tracks, GaussBlur = self.gauss, Cheat=self.Cheat, scanning_radius=self.R_scan, 
                                                                                                                    undirectedEdges=self.undirected, Dense = self.Dense)

        return Graph(x=nodes_list, a=adjacency_list, e=edge_features, y=node_labels)


#generate_3D(25, GaussBlur = 0, Cheat=False, scanning_radius=0.01, undirectedEdges=True, Dense = False)

#test = getData(1, n_min =10, n_max = 25, Gauss=0.1, Cheat=False, Save = False, Dense = True, undirected=True, newpath = "Datasets\GaussBlur")
