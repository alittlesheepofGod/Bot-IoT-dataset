import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# training dataset 
data = pd.read_csv("/mnt/d/project/dataset/Bot-Iot/Bot-Iot/BoT-IoT Dataset/DDoS/DDoS_HTTP/DDoS_HTTP[1].csv")
data.head()

# General Information about the data 
data.info()


