import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:DeEJNEAhy@34.75.124.150/postgres')
df = pd.read_sql("select * from train_labels", engine)

