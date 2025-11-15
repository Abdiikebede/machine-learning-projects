import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names=["sepal_length","sepal_width","petal_length","petal_width","class"]
data=pd.read_csv(url,names=column_names)

data.head()
data.describe()
sns.pairplot(data,hue="class")

# ✅ Define features (X) and labels (y)
x = data.drop("class", axis=1)
y = data["class"]

# ✅ Correct train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred=knn.predict(x_test)

print("Accuracy",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

x_test.head(2)

new_data = pd.DataFrame({
    "sepal_length": [5.1],
    "sepal_width": [3.5],
    "petal_length": [1.4],
    "petal_width": [0.2]
})

prediction = knn.predict(new_data)
prediction[0]
