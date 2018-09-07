import pandas
import numpy

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

file = open("male.txt",'w')
file.write("{0} {1}".format(data['Sex'].value_counts()[0],data['Sex'].value_counts()[1]))

file = open('survive.txt', 'w')
file.write('{0}'.format(round(100*data["Survived"].value_counts()[1]/data['Survived'].size,2)))

file = open('class.txt', 'w')
file.write('{0}'.format(round(100*data['Pclass'].value_counts()[1]/data.size,2)))

ages = data['Age'].values
delete = []
for i in range(ages.size):
    if not pandas.notnull(ages[i]):
        delete.append(i)
ages = numpy.delete(ages,delete)
file = open('age.txt', 'w')
file.write('{0} {1}'.format(ages.mean(), numpy.median(ages)))

file = open('pearson.txt', 'w')
file.write('{0}'.format(data['SibSp'].corr(data['Parch'],method='pearson')))

all = {}
sexs = data['Sex'].values
names = data['Name'].values
for a in range(len(names)):
    all[names[a]] = sexs[a]
womans = []
for k in all:
    if all[k] == 'female':
        womans.append(k.split(',')[1].strip())
m = []
for w in womans:
    if 'Miss' in w:
        w = w.replace('Miss. ', '')
    elif 'Mrs' in w:
        w = w.replace('Mrs. ', '')
        if '(' in w: w = w.split('(')[1]
    m.append(w.split(' ')[0])
res = pandas.value_counts(m)
print(res.index[0])
file = open("name.txt",'w')
file.write(res.index[0])

