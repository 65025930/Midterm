from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

File_Path = 'C:/Users/User/Desktop/65025930/'
File_Name = 'car_data.csv'
df = pd.read_csv(File_Path + File_Name)

df.drop(columns=['User ID'], inplace=True)
encoders =  []

for  i in range(0, len(df.columns) - 1):
    enc = LabelEncoder()
    df.iloc[:, i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
    
x = df.iloc[: 0:3]
y = df['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)

x_pred = ['Male',30,42000]

for i in range(0, len(df.columns) - 1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
    
x_pred[i] = np.array(x_pred).reshape(-1,3)

feature = x.columns.tolist()
Data_class = y .tolist()

plt.Figure(figsize=(25,20))
_= plot_tree(model,
             feature_names=feature,
             class_names=Data_class,
             labal = 'all',
             impurity = True,
             precision = 3,
             filled=True,
             rounded=True,
             fonsize=16,)
plt.show()

import seaborn as sns
feature_importances = model.feature_importances_
feature_names = ['Gender','Age','AnnualSalary']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x=feature_importances, y= feature_names)

print(feature_importances)
