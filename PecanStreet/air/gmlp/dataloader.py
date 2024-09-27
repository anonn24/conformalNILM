import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
jax.config.update("jax_default_dtype_bits", '16')

# def array(args):
#     return np.array(args, dtype = 'float16')

def data_preprocess(only_positive = False):

    # dtypes = {col: np.float16 for col in pd.read_csv('/home/interns/dhruv/active_learning/1minute_data_austin.csv', nrows=0).columns}
    # dtypes['localminute'] = str
    # dtypes['dataid'] = np.int16/home/dhruv.patel/active_learning/1minute_data_austin.csv
    data = pd.read_csv('/home/dhruv.patel/active_learning/1minute_data_austin.csv')
    houses = data['dataid'].unique()
    houses = list(houses)
    a = [9922, 6139, 3039, 4031, 4767]
    for i in a:
        houses.remove(i)
    appliances = ['clotheswasher1', 'dishwasher1', 'air1', 'refrigerator1', 'furnace1']
    required_cols = ['dataid', 'localminute']
    data_new = data[(appliances+required_cols)]
    data_new = data_new.dropna()
    if only_positive:
        numeric_cols = data_new.select_dtypes(include='number').columns
        for col in numeric_cols:
            data_new = data_new.loc[(data_new[col] >= 0) | data_new[col].isna()]
        data_aggregated = data_new.copy()
        data_aggregated['aggr'] = data_aggregated['clotheswasher1']+ data_aggregated['dishwasher1']+ data_aggregated['air1'] + data_aggregated['refrigerator1'] + data_aggregated['furnace1']
    else:
        data_aggregated = data_new.copy()
        data_aggregated['aggr'] = data_aggregated['clotheswasher1']+ data_aggregated['dishwasher1']+ data_aggregated['air1'] + data_aggregated['refrigerator1'] + data_aggregated['furnace1']
    houses = data_aggregated['dataid'].unique()
    return data_aggregated, houses

def dataloader(appliance, train, start_date, end_date,n):
    x_train =[]
    y_train = []
    n=n
    units_to_pad = n//2
    start_date = start_date
    end_date= end_date
    train_df = train[["localminute","aggr",appliance+'1']]
    train_df = train_df[(train_df["localminute"] > start_date) & (train_df["localminute"] < end_date)]
    x = train_df["aggr"].values*1000
    y_train = train_df[appliance+'1'].values*1000
    x_train = np.pad(np.array(x), (units_to_pad, units_to_pad), 'constant', constant_values = (0,0))
    x_train = np.array([np.array(x_train[i: i + n]) for i in range(len(np.array(x_train)) - n + 1)])
    x_train = np.array(x_train)    
    y_train = np.array(y_train).reshape(-1,1)
    return x_train, y_train