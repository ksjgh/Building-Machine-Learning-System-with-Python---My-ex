#-*- coding: utf-8 -*-

# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
target = data['target']
target_names = data['target_names']
labels = target_names[target]
plength = features[:, 2]

# setosa 속성을 구하기 위해 numpy 연산자를 사용, 불 배열 생성함
is_setosa = (labels == 'setosa')

max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()

print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))
