import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from blue.featurelist import FeatureList
from blue.pandas_utils import get_columns_in_df
from blue.estimators import HyperoptEstimator

from evaluation import roc_auc_truncated

train_file = './data/training.csv'
test_file = './data/test.csv'
flist = FeatureList(train_file, spec='features.yml', derived_list=None)

df_train = pd.read_csv(train_file, index_col='id')
df_train = get_columns_in_df(df_train, flist.universe)

df_test = pd.read_csv(test_file)
df_test = get_columns_in_df(df_test, flist.predictors)

hpest = HyperoptEstimator(RandomForestClassifier, max_evals=5, n_jobs=3, metric=lambda x,y : - roc_auc_truncated(x,y))
hpest.fit(df_train[flist.predictors].values, df_train[flist.target].values)
