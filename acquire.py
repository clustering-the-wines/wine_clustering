import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------

def get_wine():
    
    df = pd.read_csv('wine_quality.csv')
    
    df = df.rename(columns= {'fixed acidity': 'fixed_acidity',
             'volatile acidity': 'volatile_acidity',
             'citric acid': 'citric_acid',
             'residual sugar': 'residual_sugar',
             'free sulfur dioxide': 'free_sulphur_dioxide',
             'total sulfur dioxide': 'total_sulphur_dioxide'})
    
    df = df.drop(columns= 'Unnamed: 0')
                 
    return df

#-----------------------------------------------------------
