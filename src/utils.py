import pandas as pd
import numpy as np
import glob
import os,fnmatch
from collections import Counter
from difflib import SequenceMatcher

def findFiles(pattern, path):
    result = []
    files = glob.glob(os.path.join(path,pattern), recursive=True)
    for name in files:
        if name not in result:
            result.append(name)
    return result


def str_intersection(s1,s2):
    s = ''
    m = SequenceMatcher(None, s1, s2)
    for match in m.get_matching_blocks():
        i,j,n = match
        if n>0: # We don't want to match single chars.
            s += s1[i:i+n]
    return s


def isnan(a):
    try:
        float(a)
    except ValueError:
        return False
    return np.isnan(float(a))

def isint(a):

    if(isinstance(a,int) or isinstance(a,np.int64)):
        return True
    if isinstance(a,float):
        return False
    try:
        int(a)
        if '_' or '.' in str(a) or ',' in str(a):
            return False
        return True
    except ValueError:
        return False

def isfloat(a):

    if(isinstance(a,float)):
        return True
    if(isinstance(a,int) or isinstance(a,np.int64)):
        return False
    try:
        float(a)
        if not('.' in a or ',' in a):
            return False
        return True
    except ValueError:
        return False



def firstNotNan(df):
    df = np.array(df).reshape(-1,)
    i = 0
    if(not isinstance(df,pd.DataFrame)):
        for i in range(len(df)):
            if str(df[i]) != 'nan': #or not(np.isnan(df[i])):
                break
    return i

def notNanProportion(df):
    c = 0
    for i in df:
        if(not isnan(i)):
            c += 1
    return c/len(df)


def isnum(a):
    return isint(a) or isfloat(a)

def gini_impurity(Y):
    distribution = Counter(Y)
    s = 0.0
    size_T = len(Y)
    for Ti, size_Ti in distribution.items():
        s += ((size_Ti/size_T)**2)

    return 1 - s


def gini(y,ys):
    s = 0.0
    size_T = len(y)
    for element in ys:
        s += (len(element)/size_T) * gini_impurity(element)
    return gini_impurity(y) - s

def entropy(Y):

    distribution = Counter(Y)
    s = 0.0
    size_T = len(Y)
    for Ti, size_Ti in distribution.items():
        p = (size_Ti/size_T)
        s += (p)*np.log2(p)
    return -s

def intrinsic_value(y,ys):
    s = 0.0
    size_T = len(y)
    for Ti in ys:
        p = (len(Ti)/size_T)
        s += p * np.log2(p)
    return -s

def information_gain(y, ys):

    s = 0

    for element in ys:
        s += entropy(element)*len(element)


    return entropy(y) - s/len(y) #(entropy(y_true)*len(y_true) + entropy(y_false)*len(y_false))/len(y)
def optimized_information_gain(ys):

    s = 0

    for element in ys:
        s += entropy(element)*len(element)

    return - s


def gain_ratio(y,ys,y_with_nan):

    nan = list(y_with_nan)
    for k in y:
        nan.remove(k)
    if(len(nan) == 0):
        ys_with_nan = ys
    else:
        ys_with_nan = list(ys) + [nan]

    #if(np.isnan(information_gain(y,ys)/ intrinsic_value(y_with_nan,ys_with_nan))):
        #print(ys_with_nan)
        #print(intrinsic_value(y_with_nan,ys_with_nan))

    return information_gain(y,ys)/ intrinsic_value (y_with_nan,ys_with_nan)

def split(X,y,feature_index,values):
    if(isnum(X[0,feature_index])):
        return split_num(X,y,feature_index,values[0])
    else:
        return split_categ(X,y,feature_index,values)


def split_num(X, y, feature_index, threshold):

    X_true = []
    y_true = []
    X_false = []
    y_false = []
    true = {}
    false = {}
    for j in range(len(y)):
        if X[j][feature_index] <= threshold:
            true[j] = 1
            X_true.append(list(X[j]))
            y_true.append(y[j])
        else:
            false[j] = 1
            X_false.append(list(X[j]))
            y_false.append(y[j])

    #X_true = np.array(X_true)
    #y_true = np.array(y_true)
    #X_false = np.array(X_false)
    #y_false = np.array(y_false)

    return [X_true,X_false], [y_true,y_false],[true,false]
    #return X_true,y_true,X_false,y_false,true, false
    #return X_true, y_true, X_false, y_false

def split_categ(X,y,feature_index,values):

    Xs = []
    ys = []
    ds = []

    for value in values:
        Xs.append([])
        ys.append([])
        ds.append({})

    for j in range(len(y)):
        i = values.index(str(X[j][feature_index]))
        Xs[i].append(list((X[j])))
        ys[i].append(y[j])
        (ds[i])[j] = 1

    # for j in range(len(ys)):
    #     Xs[j] = np.array(Xs[j])#.reshape(-1,X.shape[1])
    #     ys[j] = np.array(ys[j])#.reshape(-1)

    #ys = np.array(ys)
    #Xs = np.array(Xs).reshape(len(ys),-1)
    return Xs,ys,ds


def prob_info_gain(X,y):
    v = []
    for feature_index in range(X.shape[1]):
        best_entrpy = np.float('inf')
        not_nan_rows = [a for a in range(X.shape[0]) if not isnan(X[:,feature_index][a])]

        Xnotnan = (X[not_nan_rows,:])
        ynotnan = y[not_nan_rows]
        if(len(ynotnan) == 0):
            continue
        if(isnum(Xnotnan[0,feature_index])):
            values = sorted(set(Xnotnan[:, feature_index]))
            for j in range(len(values) - 1):
                #threshold = (values[j] + values[j+1])/2
                value = (values[j] + values[j+1])/2
                [X_true, X_false], [y_true, y_false] ,[t,f] = split_num(Xnotnan, ynotnan, feature_index, value)#threshold)
                #y_false = (y_false[np.where([str(a) != 'nan' for a in X_false[:,feature_index]])[0]])


                entrpy = gini(ynotnan,[y_true,y_false])#(entropy(y_true)+entropy(y_false))#information_gain(y, y_true, y_false)
                # if(entrpy == 0):
                #     print(feature_index)
                if entrpy < best_entrpy:
                    best_entrpy = entrpy
        else:
            #values = set(np.concatenate([X[:, feature_index],['NAIA','NINA']])).
            values = sorted(set(Xnotnan[:,feature_index]))
            #values.discard(np.nan)
            #values = sorted(list(values) + ['nan'])
            Xs,ys,d = split_categ(Xnotnan,ynotnan,feature_index,values)
            if np.any(len(k) < 3 for k in ys):
                continue
            #entrpy = sum(list(entropy(k) for k in ys))
            entrpy = gini(ynotnan,ys)
            if entrpy < best_entrpy:
                best_entrpy = entrpy

        if(best_entrpy == 0):
            v.append(1)
        else:
            v.append(1/best_entrpy)

    s = 0
    for element in v:
        if element != 1:
            s += 1
    #e = []
    for i in range(len(v)):
        if v[i] != 1:
            v[i] = v[i] / s
        #else:
            #e.append(i)
    return np.array(v)#, np.array(e)#np.concatenate( v / sum(v))




# based on answer provided at: https://stackoverflow.com/questions/1566936/easy-pretty-printing-of-floats-in-python
class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self
