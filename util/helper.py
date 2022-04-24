# random state
random_state = 4995

# Data Manipulation
import numpy as np
import pandas as pd
import pickle
import json
import re
pd.options.mode.chained_assignment = None 

# System Basics
import os
import math
import datetime
import subprocess
from time import time
from tqdm import tqdm
from datetime import timedelta, date
from collections import Counter

# Math
## Stat
from scipy import stats

## Compute Distance
from geopy import distance

# Warning
import warnings
warnings.filterwarnings("ignore")

# Visualiztion
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from IPython.display import display_html, Image

sns.set_context('paper', font_scale=2)
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize': (14, 8)})

# Preprocessing
## Column Transformation
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from category_encoders import TargetEncoder
from sklearn.base import TransformerMixin

## Missing value
import sys
import sklearn.neighbors._base
import missingno as msno
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
def count_missing(df):
    msno.matrix(df)
    plt.show()
    total = df.isna().sum().sort_values(ascending=False)
    percent = (df.isna().sum()/df.isna().count()).sort_values(ascending=False)
    missing_data_info = (pd.concat([total, percent, df.dtypes], axis=1, keys=['# of missing value', '% of missing value', 'data type']).
                            to_string(formatters={'% of missing value': '{:,.2%}'.format}))
    print(missing_data_info)
    return

## Daypart
def daypart(hour):
    if hour in [2,3,4,5]:
        return "dawn"
    elif hour in [6,7,8,9]:
        return "morning"
    elif hour in [10,11,12,13]:
        return "noon"
    elif hour in [14,15,16,17]:
        return "afternoon"
    elif hour in [18,19,20,21]:
        return "evening"
    else: return "midnight"
    
## Holiday
def is_holiday(d):
    if d.weekday() > 4 or d==pd.Timestamp(2017, 11, 24, 0, 0, 0):
        return 1
    return 0

# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance

## AUC-ROC and Precision Plots
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import plot_roc_curve
from str2bool import str2bool

## Modelling tools
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold

## imbalance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

## Calibration
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

## Evaluating
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, plot_precision_recall_curve


def mem_size(size_bytes):
    '''
    Convert a file size from B to proper format
    '''
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def reduce_mem_usage(df):
    '''
    Reduce memory cost of a dataframe
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = sum(df.memory_usage())
    print('{:-^55}'.format('Begin downsizing'))
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                    convert_to = "int8"
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16) 
                    convert_to = "int16"
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    convert_to = "int32"
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
                    convert_to = "int64"
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                    convert_to = "float16"
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    convert_to = "float32"
                else:
                    df[col] = df[col].astype(np.float64)
                    convert_to = "float64"
            print(col, "converted from", col_type, "to", convert_to)
    end_mem = sum(df.memory_usage())
    print('{:-^55}'.format('Result'))
    print(f' -> Mem. usage decreased from {mem_size(start_mem)} to {mem_size(end_mem)}')
    print('{:-^55}'.format('Finish downsizing'))
    return df

def summary_memory(df):
    '''
    Calculate the memory cost of each column of a dataframe
    '''
    res = pd.concat([pd.DataFrame(df.memory_usage()).iloc[1:,:].rename({0: 'Memory'}, axis='columns'),
                     pd.DataFrame(df.dtypes).rename({0: 'Data Type'}, axis='columns')],
                    axis=1).reset_index().rename({"index": 'Veriable'}, axis='columns')
    return res


sns.set(rc={'figure.figsize':(12, 6)});
sns.set_style('darkgrid')
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.5f}'.format)

