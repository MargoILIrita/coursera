import pandas

from sklearn.tree import DecisionTreeClassifier
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
    x.append([int(ages[i]), pclasses[i], fares[i], sexs[i]])
classifier = DecisionTreeClassifier()
classifier.random_state = 241
classifier.fit(x,survived)
feature = classifier.feature_importances_
res = {feature[0]:'Age', feature[1]:'Pclass', feature[2]:'Fare',  feature[3]:'Sex'}
feature.sort()

f = open('output/tree.txt', 'w')
f.write(res[feature[2]] + " " + res[feature[3]])