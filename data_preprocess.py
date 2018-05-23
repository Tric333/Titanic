# -*- coding:utf-8 -*-
import pandas as pd

def factorize_process(obj, name):
    if isinstance(name, str):
        obj[name] = pd.factorize(obj[name])[0]
    else:
        for i in name:
            obj[i] = pd.factorize(obj[i])[0]
    return obj

def dummies_process(obj, name):
    if isinstance(name, str):
        obj = pd.concat([obj, pd.get_dummies(obj[name], prefix=name)], axis='columns')
    else:
        for i in name:
            obj = pd.concat([obj, pd.get_dummies(obj[i], prefix=i)], axis='columns')
    return obj

