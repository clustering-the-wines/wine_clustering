import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from scipy import stats

import prepare
import acquire

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)

seed = 42

#---------------------------------------------------------

def inertial_dampening(df, cols, num=11):
    
    inertia = []
    seed = 42

    for n in range(1, num):
    
        kmeans = KMeans(n_clusters=n, random_state=seed)
    
        kmeans.fit(df[cols])
    
        inertia.append(kmeans.inertia_)
        
    results_df = pd.DataFrame({'n_clusters': list(range(1,num)),
              'inertia': inertia})

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.relplot(data=results_df, x='n_clusters', y='inertia', kind='line', marker='x')

    plt.xticks(range(1, num))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Change in inertia as number of clusters increase')
    plt.show()
    
#---------------------------------------------------------

def xy_subsets(train, validate, test, target):
    '''
    This function will separate each of my subsets for the dataset (train, validate, and test) and split them further into my x and y subsets for modeling. When running this, be sure to assign each of the six variables in the proper order, otherwise it will almost certainly mess up. (X_train, y_train, X_validate, y_validate, X_test, y_test).
    '''  
    seed = 42
    
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

#---------------------------------------------------------


#---------------------------------------------------------


#---------------------------------------------------------


#---------------------------------------------------------


#---------------------------------------------------------
