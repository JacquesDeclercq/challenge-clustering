Challenge-Clustering <a name="TOP"></a>
: Clustering Faulty Bearings Project
===================

# The Mission
We were previously assigned to make an automated bearing testing system.
From the raw data, we extracted features to use it for classification.
Now we want to cluster the faulty bearings with the same characteristics to have a predicting approach on repairment.
[Go to the Previous Repo Project Web Site](https://github.com/JacquesDeclercq/challenge-classification)

# The Dataset
As the Dataset is quite big you can download it from this link
[Dataset on Kaggle](https://www.kaggle.com/isaienkov/bearing-classification?select=bearing_signals.csv)

# Libraries to Import
#Data analysis libraries
import numpy as np 
import pandas as pd 

#Visulization and statistics libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from scipy import fftpack
import seaborn as sns
style.use('seaborn')

# Model related libraries
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.vq import kmeans, vq
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

## Features Used
-------------------------------------------------------------------------------------------------------------------------
The 9 extracted features out of the raw data were :

1. Max - acceleration on x,y,z axis, RPM, HZ, W
2. Min - acceleration on x,y,z axis, RPM, HZ, W
3. Mean - acceleration on x,y,z axis, RPM, HZ, W
4. STD - acceleration on x,y,z axis, RPM, HZ, W
5. Median - acceleration on x,y,z axis, RPM, HZ, W
6. Range - acceleration on x,y,z axis, RPM, HZ, W
7. Kurtosis - acceleration on x,y,z axis, RPM, HZ, W
8. Skew - acceleration on x,y,z axis, RPM, HZ, W
9. Var - acceleration on x,y,z axis, RPM, HZ, W

#Normalizing & Standardizing The Data
1. from sklearn import StandarScaler
2. from scipy.cluster.vq import whiten

# PCA

#PairPlot

<img src="https://github.com/JacquesDeclercq/challenge-clustering/blob/main/images/Screenshot%202021-08-11%20at%2016.09.53.png" width="350">

# How many clusters ?
To find the amount of clusters we need to use we plotted through the Elbow, Dendogram & Intertia plot

# Elbow Plot
#Scaled Data
<img src="https://github.com/JacquesDeclercq/challenge-clustering/blob/main/images/Screenshot%202021-08-11%20at%2016.14.06.png" width="350">
#Non Scaled Data
<img src="https://github.com/JacquesDeclercq/challenge-clustering/blob/main/images/Screenshot%202021-08-11%20at%2016.14.06.png" width="350">

# Inertia Plot

<img src="https://github.com/JacquesDeclercq/challenge-clustering/blob/main/images/Screenshot%202021-08-11%20at%2016.14.06.png" width="350">

# Dendogram

<img src="https://github.com/JacquesDeclercq/challenge-clustering/blob/main/images/Screenshot%202021-08-11%20at%2016.14.29.png" width="350">

# Silhouette Score

See video

# Best Features & Plots

Max Acc A2_x
STD A2_x
FFT A2_x

# Clustering Techniques

K-Means
Birch
Hierarchical
DBSCAN
Agglomerative Clustering
Affinity Propagation

