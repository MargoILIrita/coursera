import pandas

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

pclasses = data['Pclass'].values
fares = data['Fare'].values
ages = data['Age'].values
sexs = data['Sex'].values
survived = data['Survived'].values

delete = []
for i in range(ages.size):
    if not pandas.notnull(ages[i]):
        delete.append(i)
ages = np.delete(ages,delete)
pclasses = np.delete(pclasses, delete)
fares = np.delete(fares, delete)
sexs = np.delete(sexs, delete)
survived = np.delete(survived, delete)


for i in range(sexs.size):
    if sexs[i] == 'female':
        sexs[i] = 0
    else:
        sexs[i] = 1

x = []
for i in range(ages.size):
    x.append([ages[i], pclasses[i], fares[i], sexs[i]])
