import csv
import numpy as np
import sklearn.metrics


def sigma(w, xarr):
    return [1/(1 + np.exp(-1*w[0]*x[0]-w[1]*x[1]))
            for x in xarr]

def commonpart(x,y,w):
    return 1 - (1/(1 + np.exp(-1*y*np.dot(w,x))))


def w1U(x,y,w,k):
    l = len(y)
    sum = 0
    for i in range(l):
        sum+=y[i]*x[i][0]*commonpart(x[i],y[i],w)
    sum*=k/l
    return w[0] + sum


def w2U(x,y,w,k):
    l = len(y)
    sum = 0
    for i in range(l):
        sum += y[i] * x[i][1] * commonpart(x[i], y[i], w)
    sum *= k / l
    return w[1] + sum


def w1L(x,y,w,k,C):
    return w1U(x,y,w,k) - k*C*w[0]


def w2L(x,y,w,k,C):
    return w1U(x,y,w,k) - k*C*w[1]


def wU(w0,x,y,k=0.1):
    return np.array([w1U(x,y,w0,k), w2U(x,y,w0,k)])


def wL(w0,x,y,k=0.1,C=10):
    return np.array([w1L(x,y,w0,k,C),w2L(x,y,w0,k,C)])


def gardient(foo, w0, x, y):
    i = 0
    for i in range(10000):
        w = foo(w0,x,y)
        if (np.linalg.norm(w0 - w) <= 10**(-5)):
            print(i)
            return w
        w0 = w
    print(i)
    return w0


reader = csv.reader(open('data-logistic.csv'),delimiter=',')
data = []
y = []
for row in reader:
    data.append([float(row[1]), float(row[2])])
    y.append(float(row[0]))

wl = gardient(wL,np.zeros(2),data,y)
wu = gardient(wU,np.zeros(2),data,y)
print(wl.__str__() + " " + wu.__str__())

resL = sklearn.metrics.roc_auc_score(y,sigma(wl,data))
resU = sklearn.metrics.roc_auc_score(y,sigma(wu,data))
print(resU)
print(resL)

f = open("output/8answer.txt", 'w')
f.write("{0} {1}".format(round(resU,3), round(resL,3)))