from umap import UMAP
import pandas as pd
import numpy as np
X = pd.read_csv('../clean_data/X.csv')
y = pd.read_csv('../clean_data/y.csv')
reducer = UMAP()
X_umaped = reducer.fit_transform(X)
pd.DataFrame(X_umaped).to_csv('../clean_data/X_umaped.csv', index=False)
