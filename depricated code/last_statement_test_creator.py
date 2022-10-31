import pandas as pd

# import required module
import os
# assign directory
directory = r'.\amex-default-prediction\test_data_split'

df_combined = pd.DataFrame()
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print("Working on " + f)
        df_last_statements = pd.DataFrame()
        df_test_data_chunk = pd.read_csv(f)
        df_test_data_chunk = df_test_data_chunk.groupby("customer_ID").last()
        df_combined = pd.concat([df_combined,df_test_data_chunk])

df_combined = df_combined.groupby("customer_ID").last() #because of overlap between split files, take final DF, group by again, and take last
df_combined.to_csv(r".\amex-default-prediction\test_data_last_statement.csv")
