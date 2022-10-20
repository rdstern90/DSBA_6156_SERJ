import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import exc
import os

working_directory = os.path.dirname(os.path.realpath(__file__))
engine = create_engine('postgresql://postgres:*****@34.75.124.150/postgres')

train_labels_relative_path = r'amex-default-prediction\train_labels.csv'

train_labels = pd.read_csv(os.path.join(working_directory,train_labels_relative_path))
train_labels.columns= train_labels.columns.str.lower() #we want lowerdcase column names so postgresql doesn't make us to use double quotes

tableName = "train_labels_all"
train_labels.to_sql(tableName, engine, if_exists='replace', index=False, chunksize=100000)
