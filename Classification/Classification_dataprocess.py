from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
X = wine.data.features 
yTemp = wine.data.targets 
X, yTemp = shuffle(X, yTemp, random_state=42)  # random_state for reproducibility
yTemp = yTemp.reset_index(drop=True)
X = X.reset_index(drop=True)

# Assuming X and yTemp are your features and targets, respectively


# Normalize the features
X_normalized_df = (X - X.mean()) / X.std()

pd.set_option('display.max_rows', None)

rows = []

for i in range(yTemp.shape[0]):
    if yTemp.iloc[i, 0] == 1:
        rows.append([1, 0, 0])
    elif yTemp.iloc[i, 0] == 2:
        rows.append([0, 1, 0])
    elif yTemp.iloc[i, 0] == 3:
        rows.append([0, 0, 1])


y = pd.DataFrame(rows, columns=['Class1', 'Class2', 'Class3'])



X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.2, random_state=42)
