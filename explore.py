import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

import prepare
import acquire

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)

seed = 42

#------------------------------------------------------------

def mannwhitney_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform a mannwhitneyu test on them and print out the t-statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    t, p = stats.mannwhitneyu(group1, group2)

    alpha = .05
    seed = 42

    print(f'T-statistic = {t:.4f}') 
    print(f'p-value     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def ind_ttest_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform an independent t-test on them and print out the t-statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    t, p = stats.ttest_ind(group1, group2, equal_var=False)

    alpha = .05
    seed = 42

    print(f'T-statistic = {t:.4f}') 
    print(f'p-value     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def pearsonr_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform a pearson r test on them and print out the t-statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    t, p = stats.pearsonr(group1, group2)

    alpha = .05
    seed = 42

    print(f'T-statistic = {t:.4f}') 
    print(f'p-value     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def spearmanr_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform a spearman r test on them and print out the t-statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    t, p = stats.spearmanr(group1, group2)

    alpha = .05
    seed = 42

    print(f'T-statistic = {t:.4f}') 
    print(f'p-value     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------


#------------------------------------------------------------


#------------------------------------------------------------


#------------------------------------------------------------


#------------------------------------------------------------


#------------------------------------------------------------