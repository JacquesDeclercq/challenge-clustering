Challenge-Clustering <a name="TOP"></a>
Clustering Faulty Bearings Project
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
Max - acceleration on x,y,z axis, RPM, HZ, W
Min - acceleration on x,y,z axis, RPM, HZ, W
Mean - acceleration on x,y,z axis, RPM, HZ, W
STD - acceleration on x,y,z axis, RPM, HZ, W
Median - acceleration on x,y,z axis, RPM, HZ, W
Range - acceleration on x,y,z axis, RPM, HZ, W
Kurtosis - acceleration on x,y,z axis, RPM, HZ, W
Skew - acceleration on x,y,z axis, RPM, HZ, W
Var - acceleration on x,y,z axis, RPM, HZ, W
