
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import tensorflow as tf

print(tf.__version__)
print(np.__version__)
tf.config.list_physical_devices('GPU')

from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, AUC


from spektral.data import DisjointLoader
from spektral.layers import GCSConv, ECCConv, AGNNConv, GeneralConv, GCNConv

from spektral.transforms.normalize_adj import NormalizeAdj

from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve

from make_dataset import getData

class GNN(Model):

    def __init__(self, N_hidden_layers, N_class_labels = 2, n_dimensions = 3):
        super().__init__()

        #defining model layers:

        self.graph_conv1 = GCSConv(N_hidden_layers) # Convolutional layers with trainable skip - the 'default' GNN layer type(?)
        self.graph_conv2 = GCSConv(N_hidden_layers)
        self.graph_conv3 = GCSConv(N_hidden_layers)

        self.gcn1 = GCNConv(N_hidden_layers)
        self.gcn2 = GCNConv(N_hidden_layers)

        self.agnn1 = AGNNConv(trainable =True,aggregate='sum')
        self.agnn2 = AGNNConv(trainable =True,aggregate='sum')

        self.leaky = LeakyReLU(alpha=0.2) #rectified linear activation for hidden training layers

        self.ecc = ECCConv(64)
        self.ecc2 = ECCConv(N_hidden_layers)

        self.general = GeneralConv(N_hidden_layers, dropout=0.5)
        self.general2 = GeneralConv(N_hidden_layers)

        self.dense_clas = Dense(N_class_labels, 'softmax')  # Classification layer - for determining if a track belongs to the displaced vertex


    def call(self, inputs): #forward propogation of the 

        x, a, e, i= inputs

        # ECC

        #H = self.ecc([x,a,e])
        #H = self.leaky(H)

        #2xGCN

        # H = self.gcn1([x,a])
        # H = self.leaky(H)
        # H = self.gcn2([H,a])
        # H = self.leaky(H)

        # 2xGCS - Best performing architecture

        H = self.graph_conv1([x,a])
        H = self.leaky(H)

        H = self.graph_conv2([H,a])
        H = self.leaky(H)

        # H = self.agnn1([H,a])
        # H = self.leaky(H)

        # GeneralCONV        

        # H = self.general([x,a])
        
        # H = self.leaky(H)
        # H = self.general2([H,a])
        # H = self.leaky(H)

        #classification on the nodes

        classes = self.dense_clas(H) #Output layer

        return classes


@tf.function(experimental_relax_shapes=True) #functionality: 'When True, tf.function may generate fewer, graphs that are less specialized on input shapes.' - this may be the cause of cmdline errors?
def train(inputs, target):
    with tf.GradientTape(persistent=True) as tape:

        predictions= model(inputs,training=True)

        loss = loss_scc(target, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def query(inputs, target):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(inputs, training=False)

        AUC_metric.update_state(target, predictions[:,1])

        loss = loss_scc(target, predictions)
        
    return predictions, target, loss

def rocCurve(truth_vals, predictions): #Generates ROC Curve values for the positive class for a given set of predictions and labels
    fp, tp, _ = roc_curve(truth_vals[1], predictions[:,1], drop_intermediate=False)
    return fp,tp

def confusionMatrix(truth, predictions): 

    return confusion_matrix(truth[1], np.where(predictions[:,1] > 0.5, 1, 0))  

def precisionRecall(truth, predictions):

    return precision_recall_curve(truth[1], predictions[:,1], pos_label=1)


losses = np.empty(0)
AUC_list = np.empty(0)
epoch_list = np.empty(0)
validation_list = np.empty(0)
save_epoch = np.empty(0)

### MODEL PARAMETERS ###

save_i = 1000
update_loss = 100
File_Name = str(sys.argv[1])
metrics_names = ['loss']
graph = True #whether to output data

N_channels = 64
N_epoch = 40000
n_tests = 300
learning_rate=1e-2

optimiser = Adam(learning_rate, beta_1=0.9, decay=0, amsgrad=True) #SGD(learning_rate, momentum=0.9)#
loss_scc = SparseCategoricalCrossentropy()
loss_mse = MeanSquaredError()
loss_cc = CategoricalCrossentropy()
AUC_metric = AUC()

### DATASET PARAMETERS ###

savepath = "Datasets\Dense" # Search path for the dataset. If a dataset with the specific graph parameters
                            # and number of graphs does not exist at this location, the code will generate them.
                            # WARNING - THIS CAN BE VERY LARGE (~10GB)
N_graphs = 25000

#Graph Parameters

n_min = 10
n_max = 25
Gauss = 0   # Std. deviation of gaussian noise added to y-z position of graph hits
Cheat = False # Whether to feed tracks via adjacency matrix
R_scan = 0.01 #Scanning radius for crude Kalman filter

DenseAdj = True #Specifies fully connected graphs between detector planes
unDirectedAdj = True # Specifies symmetric adjacency matrix (ie. undirected graph edges)

#Initialising the model

model = GNN(N_channels, N_class_labels=2, n_dimensions=3) # 2 categories for SparseCategoricalCrossentropy, 3 dimensions for the regression task

model.compile(optimizer = 'Adam', loss = tf.keras.losses.SparseCategoricalCrossentropy, metrics=[AUC_metric,TruePositives(name='tp'), 
                                                                                                 TrueNegatives(name='tn'), FalseNegatives(name='fn'),
                                                                                                 FalsePositives(name='fp')])

dataset = getData(N_graphs, n_min = n_min, n_max = n_max, Gauss=Gauss, Cheat=Cheat, R_scan=R_scan, newpath = savepath, 
                    Dense = DenseAdj, undirected=unDirectedAdj, transforms = NormalizeAdj())

loader = DisjointLoader(dataset, batch_size=100, node_level=True)
prog_bar = tf.keras.utils.Progbar(N_epoch, stateful_metrics=metrics_names)


for i, batch in enumerate(loader):

    if i > N_epoch:
        break

    A = train(*batch)

    losses = np.append(losses, A.numpy())

    if i % save_i == 0 or i == 0: # Evaluate the model on fresh data at the specified interval

        print(f'Step {i}')
        print(i,' loss: ',A.numpy())

        if i == 0: continue

        AUCtest = getData(n_tests, n_min = n_min, n_max = n_max, Gauss=Gauss, Cheat=Cheat,  R_scan=R_scan, newpath = "Datasets\Tests",Save = False,Dense = DenseAdj, undirected=unDirectedAdj, transforms = NormalizeAdj())
        AUCtestloader = DisjointLoader(AUCtest, batch_size = n_tests, node_level=True, epochs = 1)

        for j, aucbatch in enumerate(AUCtestloader):

            predictions, target, validation_loss = query(*aucbatch)
            AUC_list = np.append(AUC_list, AUC_metric.result())
            epoch_list = np.append(epoch_list,i)
            validation_list = np.append(validation_list, validation_loss)
            save_epoch = np.append(save_epoch, i)

            fp, tp = rocCurve(aucbatch, predictions)

            precision, recall, pr_thresh = precisionRecall(aucbatch, predictions)

            confusion = confusionMatrix(aucbatch, predictions)

    
    vals = [('loss',A.numpy())]
    prog_bar.update(i)
    prog_bar.add(update_loss, vals)

print(AUC_list)


if graph == True:
    fig, axs = plt.subplots(2,2)

    

    axs[0,0].plot(epoch_list,AUC_list)
    axs[1,0].plot(losses, label = 'Training Loss')
    axs[1,0].plot(save_epoch,validation_list, color ='orange', label = 'Validation Loss')
    axs[0,0].set(xlabel='epoch')
    axs[1,0].set(xlabel='epoch')
    axs[0,0].set(ylabel='AUC')
    axs[1,0].set(ylabel='Loss')
    axs[1,0].legend(loc='best')

    #ROC Curve

    axs[0,1].plot(fp,tp,color='blue', label = f'AUC = {AUC_list[-1]}')
    axs[0,1].plot([0,1], [0,1], color='red', linestyle = 'dashed')
    axs[0,1].set(xlabel = 'FPR', ylabel = 'TPR')
    axs[1,0].legend(loc='best')

    #PR Curve

    axs[1,1].plot(recall, precision)
    axs[1,1].plot([0,1], [0.5,0.5], color='red', linestyle = 'dashed')
    axs[1,1].set(xlabel = 'Recall', ylabel = 'Precision')

    fig.tight_layout()

    fig.savefig(f'Figures\{File_Name}_NG_{N_graphs}_TR{n_min}-{n_max}_Gauss{Gauss}_Scan{R_scan}_Cheat_{Cheat}_E{N_epoch}_AUC.png',  bbox_inches='tight')
    fig.show()

    #plt.close('all')

    df = pd.DataFrame({"Epoch": epoch_list, "AUC": AUC_list})

    roc = pd.DataFrame({"TPR": tp, "FPR": fp})

    pr_curve = pd.DataFrame({"Precision": precision, "Recall": recall})

    loss_final = pd.DataFrame({"Training Loss" : losses})
    
    valid_loss = pd.DataFrame({"Validation Loss": validation_list, "Validation Epoch" : save_epoch})

    df.to_csv(f"Data\AUC\{File_Name}_NG_{N_graphs}_TR{n_min}-{n_max}_Gauss{Gauss}_Scan{R_scan}_Cheat_{Cheat}_E{N_epoch}_AUC.csv", index = False)

    roc.to_csv(f"Data\ROC\{File_Name}_NG_{N_graphs}_TR{n_min}-{n_max}_Gauss{Gauss}_Scan{R_scan}_Cheat_{Cheat}_E{N_epoch}_ROCData.csv", index=False)

    pr_curve.to_csv(f"Data\PR\{File_Name}_NG_{N_graphs}_TR{n_min}-{n_max}_Gauss{Gauss}_Scan{R_scan}_Cheat_{Cheat}_E{N_epoch}_PRData.csv", index=False)
    
    loss_final.to_csv(f"Data\LOSS\Training\{File_Name}_NG_{N_graphs}_TR{n_min}-{n_max}_Gauss{Gauss}_Scan{R_scan}_Cheat_{Cheat}_E{N_epoch}TrainingLoss.csv")

    valid_loss.to_csv(f"Data\LOSS\Validation\{File_Name}_NG_{N_graphs}_TR{n_min}-{n_max}_Gauss{Gauss}_Scan{R_scan}_Cheat_{Cheat}_E{N_epoch}ValidationLoss.csv")

    model.save(f'Models\{File_Name}_CH{N_channels}_G{N_graphs}_E{N_epoch}')

quit()