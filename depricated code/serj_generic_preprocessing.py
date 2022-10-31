from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd

class ImputeAsDataFrame(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def fit_transform(self, df):
    cat_cols = df.select_dtypes(exclude="number").columns
    num_cols = df.select_dtypes(include="number").columns

    for col in cat_cols:
      df[col] = SimpleImputer(missing_values=None, strategy="most_frequent").fit_transform(df[[col]])
    for col in num_cols:
      df[col] = SimpleImputer(strategy="mean").fit_transform(df[[col]])

    if df.isna().sum().sum() > 0:
      print(f"WARNING: {df.isna().sum().sum()} nulls still exist after imputing.")
    return df

class BackToDataFrame(BaseEstimator, TransformerMixin, cols=[]):
  def fit(self, X, y=None):
    return self
  def fit_transform(self, X):
    df = pd.DataFrame(X, columns=cols)
    return df