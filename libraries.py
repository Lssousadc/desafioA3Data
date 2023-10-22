'''
Este arquivo é utilizado para colocar todas as bibliotecas utilizadas no arquivo de treino/validação do modelo
'''


import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import joblib



# setando opcoes de display

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, make_scorer, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
