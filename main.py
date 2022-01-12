from sklearn.svm import SVC
import pandas as pd
import numpy as np
from get_coords import path_to_coords

# Classification:
# 0 = t-pose
# 1 = tree-pose

df = pd.read_csv("pose_data.csv").values

X_train = df[0:, 0:-1]
y_train = df[0:, -1]

path = "tree-test/3.png"
X_test = np.array([path_to_coords(path)])


clf = SVC()
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print(preds)