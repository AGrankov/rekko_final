import numpy as np
import pandas as pd
from sklearn import preprocessing

def labeling_col(df, col):
    df[col] = df[col].fillna('na')
    le = preprocessing.LabelEncoder().fit(df[col].unique().tolist())
    df[col] = le.transform(df[col].values)
    return le

def normalize_int_col(df, col):
    scaler = preprocessing.StandardScaler()
    df[col] = scaler.fit_transform(df[col].values.reshape((-1, 1)))[:, 0]
    return scaler

def process_sequence(df, col):
    seq = df.groupby('user_uid').agg({col: (lambda x: x.values.tolist())}).values
    seq = [x[0] for x in seq]
    return seq
