# Code for Analyses - Prediction

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

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

n_iterations=int(sys.argv[1])

Prediction = pd.read_csv('/data/project/EF_prosody/superultra_final_dataNN.csv')


#EF=Prediction.loc[:,'Corsi_UBS':'psy_sw_err_incong']
EF=Prediction.loc[:,'tmt_BTA':'tmt_FB']
EFarray=np.array(EF)

Prosody=Prediction.loc[:,'F0semitoneFrom27.5Hz_sma3nz_amean_x':'equivalentSoundLevel_dBp']

total_EFtargets =6
Correlations = []
target_list = {}

for target in range(total_EFtargets):
    target_list[target] = pd.DataFrame(EFarray[:,target])
    
Prosody_dict=Prosody.to_dict()



for target, df_Complete in target_list.items():
    
    df_Complete['SEX'] = Prediction['SEX']
    df_Complete['AGE'] = Prediction['AGE']
    df_Complete['EDLEV'] = Prediction['EDLEV']
    
    
    df_Complete = df_Complete.join(pd.DataFrame(Prosody_dict , index=df_Complete.index))
    
    df_Complete = df_Complete.rename(columns=lambda col: str(col))
    
    df_Complete = df_Complete.astype('float64')
    
    X = Prosody.columns.to_list()
    y='0'
    confounds = ['AGE','SEX', 'EDLEV']
    preprocess_X = ['remove_confound']
    
    #for statification on cross validations we splitting all the targets on bins
    num_splits = 10
    num_repeats = 10
    num_bins = math.floor(len(df_Complete['0']) / num_splits)  # num of bins to be created
    bins_on = df_Complete['0']  # variable to be used for stratification
    qc = pd.cut(bins_on.tolist(), num_bins)  # divides data in bins
    df_Complete['bins'] = qc.codes
    groups = 'bins'
    rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=88)
    cv_stratified = StratifiedGroupsKFold(n_splits=num_splits, random_state=88, shuffle=True)
    rpcv10 = RepeatedKFold(n_splits=10,n_repeats=10, random_state=88)
    cv10=KFold(n_splits=10)
    cv2=KFold(n_splits=2)
    
    
    
    
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
        print(rgss)
        name = rgss.__class__.__name__
        real_scores = run_cross_validation(
            X=X, y=y, data=df_Complete, preprocess_X=preprocess_X, cv=cv10,
            problem_type='regression', model=rgss, return_estimator='cv', confounds=confounds, seed=111, scoring=['r2'], 
            model_params=dict(
                remove_confound__model_confound=[RandomForestRegressor()]), n_jobs=1
        )

        
        r2_scores = real_scores['test_r2']
        mean_r2 = r2_scores.mean()
        

        import random
        random.seed(10)
        
        
        permuted_scores = []
        
        
        df_permuted = df_Complete.copy()
        df_permuted[X] = df_permuted[X].sample(frac=1, random_state=n_iterations).values

        perm_scores = run_cross_validation(
            X=X, y=y, data=df_permuted, preprocess_X=preprocess_X, cv=cv10,
            problem_type='regression', model=rgss, return_estimator='cv',
            confounds=confounds, seed=111, scoring=['r2'], 
            model_params=dict(
                remove_confound__model_confound=[RandomForestRegressor()])
        )

        perm_r2_scores = perm_scores['test_r2']
        mean_perm_r2 = perm_r2_scores.mean()

        #permuted_scores_dict = (mean_perm_r2)
        permuted_scores.append(mean_perm_r2)
        permuted_scoresarray=np.array(permuted_scores)
            
      
        Path(f'./permutation_results/finalTRYlast_4b_regressor_{name}_target_{target}/').mkdir(parents=True, exist_ok=True)

        saved_mean_perm_r2=pd.DataFrame({'mean_r2' :[mean_perm_r2]})
        saved_mean_perm_r2.to_csv(f'./permutation_results/finalTRYlast_4b_regressor_{name}_target_{target}/saved_mean_perm_r2_{n_iterations}.csv')