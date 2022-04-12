# ProjectMSc-Code
Graph Generation and GNN Code for Final Year Project

Requires Tensorflow 2.3.0, Numpy <1.2 and Spektral


generate3D_EdgeFull.py - a Monte-Carlo simulation, generating particle tracks from a primary and displaced vertex and outputting hits where these tracks intersect detector planes.



make-dataset.py - handles graph generation from the output of generate3D

GNN_Tidied.py - main GNN code, usage 'python GNN_Tidied.py <IDENTIFIER>', where <IDENTIFIER> is 


Data output from GNN_Tidied.py requires a current working directory setup as follows:


```(CWD)
-GNN_Tidied.py
-make-dataset.py
-generate3D_EdgeFull.py

-(Figures)
-(Models)
-(Data)
-(Datasets)```
