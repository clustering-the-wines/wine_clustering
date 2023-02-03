import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

import prepare
import acquire

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)

seed = 42

#---------------------------------------------------------

def inertial_dampening(df, cols, num=11):
    '''
    This function will take in a dataframe, a list of columns as well as a number of clusters to create, fit, and append the inertia from the clustering model into a list, convert it to a dataframe, and then create a line plot of each inertia to help determine the best number of clusters.
    '''
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

def subset_df(df, stratify=None, seed=42):
    '''
    This function takes in a DataFrame and splits it into train, validate, test subsets for our modeling phase. Does not take in a stratify option.
    '''
    train, val_test = train_test_split(df, train_size=.6, random_state=seed)
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed)
        
    return train, validate, test

#---------------------------------------------------------

def xy_subsets(train, validate, test, target):
    '''
    This function will separate each of my subsets for the dataset (train, validate, and test) and split them further into my x and y subsets for modeling. When running this, be sure to assign each of the six variables in the proper order.
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

def make_baseline(df, baseline, col):
    '''
    This function is used to create a column within the dataframe to make a baseline column, and then calculate the baseline accuracy. Needs to be optimized more, but functions as is currently. Make sure to use the word 'baseline' when calling function.
    '''
    
    seed = 42
    
    df[baseline] = df[col].value_counts().idxmax()    

    base = (df[col] == df[baseline]).mean()
    
    print(f'Baseline Accuracy is: {base:.3}')
    
#---------------------------------------------------------

def rf_gen(X_train, y_train, X_validate, y_validate):
    
    metrics = []
    
    seed = 42

    for i in range(1, 20):
        '''
        This function will create a dataframe of Random Forest models of varying max_depths and 
        compare the differences from the train and validate sets and return the dataframe. 
        '''
        rf = RandomForestClassifier(max_depth=i, min_samples_leaf=3, n_estimators=200, random_state=42)
        rf = rf.fit(X_train, y_train)
        
        in_sample_accuracy = rf.score(X_train, y_train)
        out_of_sample_accuracy = rf.score(X_validate, y_validate)
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
            }
    
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    
    return df

#---------------------------------------------------------

def dectree_gen(X_train, y_train, X_validate, y_validate):
    
    metrics = []
    
    seed = 42

    for i in range(1, 20):
        '''
        This function will create a dataframe of Decision Tree models of varying max_depths and 
        compare the differences from the train and validate sets and return the dataframe. 
        '''
        dectree = DecisionTreeClassifier(max_depth=i, random_state=42)
        dectree = dectree.fit(X_train, y_train)

        in_sample_accuracy = dectree.score(X_train, y_train)
        out_of_sample_accuracy = dectree.score(X_validate, y_validate)
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
            }
    
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    
    return df
#---------------------------------------------------------

def knn_gen(X_train, y_train, X_validate, y_validate):
    
    metrics = []

    seed = 42
    
    for i in range(1, 21):
        '''
        This function will create a dataframe of KNN models of varying n_neighbors and compare the differences from 
        the train and validate sets and return the dataframe.
        '''
        knn = KNeighborsClassifier(n_neighbors=i, weights='uniform')
        knn = knn.fit(X_train, y_train)

        in_sample_accuracy = knn.score(X_train, y_train)
        out_of_sample_accuracy = knn.score(X_validate, y_validate)
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
            }
    
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    
    return df

#---------------------------------------------------------
