import cPickle as pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from blue.featurelist import FeatureList, extract_featurelist
from blue.modelers.modeler import Modeler
from evaluation import roc_auc_truncated, ModelValidator

FEATURE_LIST = 'features.yml'
DERIVED_LIST = None

TRAIN = './data/training.csv'
TEST = './data/test.csv'

flist = FeatureList(TRAIN, spec=FEATURE_LIST, derived_list=DERIVED_LIST)

def load_train(features):
    df_train = pd.read_csv(TRAIN, index_col='id')
    return extract_featurelist(df_train, features)

def load_test():
    df_test = pd.read_csv(TEST, index_col='id')
    return extract_featurelist(df_test, features)


def model(est, df):

    modeler = Modeler(est, flist.target, roc_auc_truncated, features=flist.predictors, samplepct=0.05)
    modeler.fit(df)

    validator = ModelValidator(flist)
    valid = validator.validate(modeler)
    cvscore = np.mean(modeler.scores)
    
    print valid
    print cvscore

    return modeler, valid, cvscore
 
if __name__ == '__main__':
    df_train = load_train(flist.universe)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=1234)
    rf_model, valid, cvscore = model(rf, df_train)



    
