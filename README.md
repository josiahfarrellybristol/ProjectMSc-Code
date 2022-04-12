# ProjectMSc-Code
Graph Generation and GNN Code for Final Year Project

Requires Tensorflow 2.3.0, Numpy <1.2 and Spektral


generate3D_EdgeFull.py - a Monte-Carlo simulation, generating particle tracks from a primary and displaced vertex and outputting hits where these tracks intersect detector planes.



make-dataset.py - handles graph generation from the output of generate3D

GNN.py - main GNN code, usage:

```
python GNN.py <IDENTIFIER>'
```
where IDENTIFIER is appended to the output datafiles.


Data output from GNN_Tidied.py requires a current working directory setup as follows:


```
(CWD)
-GNN_Tidied.py
-make-dataset.py
-generate3D_EdgeFull.py

-(Figures)
-(Models)
-(Data)
-(Datasets)
```

Dataset parameters are controlled within the code of GNN.py - will generate a dataset if none exists in the searched path. To change the parameters, source file must be edited - sorry!
