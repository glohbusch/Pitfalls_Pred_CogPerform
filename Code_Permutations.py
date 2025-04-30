# Code for Analyses - Permutations
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from julearn import run_cross_validation
from julearn.utils import configure_logging

from sklearn.metrics import make_scorer
from julearn.scoring import register_scorer
import scipy 

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor

import math
from julearn.model_selection import StratifiedGroupsKFold
from sklearn.model_selection import RepeatedStratifiedKFold

configure_logging(level='INFO')
from pathlib import Path

import sys



def pearson_scorer(y_true, y_pred):
        return scipy.stats.pearsonr(  # type: ignore
            y_true.squeeze(), y_pred.squeeze())[0]

    
register_scorer(scorer_name='pearsonr', scorer=make_scorer(pearson_scorer))


Prediction = pd.read_csv('/data/project/EF_prosody/superultra_final_dataNN.csv')
 


#EF=Prediction.loc[:,'Corsi_UBS':'psy_sw_err_incong']
EF=Prediction.loc[:,'tmt_BTA':'tmt_FB']
EFarray=np.array(EF)

total_EFtargets =6
Correlations = []
target_list = {}

for target in range(total_EFtargets):
    target_list[target] = pd.DataFrame(EFarray[:,target])
    


for target, df_Complete in target_list.items():
    
   
    y='0'
 
    
    Regressors = [
    RandomForestRegressor(),
    #SVR(),
    #ExtraTreesRegressor(),
    #AdaBoostRegressor(),
    #BaggingRegressor(),
    #GradientBoostingRegressor(),
    #LinearRegression(),
    #Ridge(),
    #RidgeCV(),
    #SGDRegressor(),
    ]

    log_cols = ["Regressor", "Accuracy", "P_value"]
    log = pd.DataFrame(columns=log_cols)
    p_value_dict = {}
    
    print(target)

    ####
    #exchange parameters of run_cross_validation in order to get the different results
    ####
    
    for rgss in Regressors:
        name = rgss.__class__.__name__
        print(rgss)
       
        permuted_scores = []
        mean_r2 = pd.read_csv(f'./permutation_results/verylast_4f_regressor_{name}_target_{target}/saved_mean_real_r2_0.csv').iloc[0,1]


        for n_iterations in range(1000):
            
            mean_perm_r2 = pd.read_csv(f'./permutation_results/verylast_4f_regressor_{name}_target_{target}/saved_mean_perm_r2_{n_iterations}.csv').iloc[0,1]
            permuted_scores.append(mean_perm_r2)
            permuted_scoresarray=np.array(permuted_scores)
        

        p_val = (np.sum(permuted_scoresarray > mean_r2)+1)/1001

        p_value_dict[name] = p_val

        log_entry = pd.DataFrame([[name, mean_r2, np.mean(p_val)]], columns=log_cols)
        log = pd.concat([log_entry,log], axis=0)


    log['Accuracy']=log['Accuracy'].map('{:,.3f}'.format)
    log['P_value']=log['P_value'].map('{:,.3f}'.format)
    Correlations.append({'target_EF': target, 'Model': log["Regressor"], 'Accuracy': log["Accuracy"], 'P_value': log["P_value"]})



Correlations_df = pd.DataFrame(Correlations)


np.savetxt('./permuted_scoresarray_verylast_4f_Sept23.txt', np.array(permuted_scoresarray), fmt='%.2f')
EF_names = pd.read_csv('./EF_names_TMT.csv')
EF_names = EF.columns
Correlations_df['EF_label'] = EF_names
Correlations_df = Correlations_df[['EF_label'] + Correlations_df.columns[:-1].tolist()]
Correlations_df.to_csv('/data/project/EF_prosody/pred_results/verylast_submitted_4f_withCR_CRmodelRF.csv')