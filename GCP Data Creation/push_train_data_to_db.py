import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import exc
import os

working_directory = os.path.dirname(os.path.realpath(__file__))
engine = create_engine('postgresql://postgres:****@34.75.124.150/postgres') 

train_data_relative_path = r'amex-default-prediction\train_data.csv'

train_data = pd.read_csv(os.path.join(working_directory,train_data_relative_path))
train_data.columns= train_data.columns.str.lower() #we want lowerdcase column names so postgresql doesn't make us to use double quotes

tableName = "train_data_all"
train_data.to_sql(tableName, engine, if_exists='replace', index=False, chunksize=100000)
