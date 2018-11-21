from model import NaiveBayesClassifier
import numpy as np


X = np.array([[1, 0, 1],
              [1, 1, 1],
              [1, 1, 0],
              [0, 0, 1],
              [0, 1, 1]])
y = np.array([[0, 1],
              [0, 1],
              [0, 1],
              [1, 0],
              [1, 0]])

t = np.array([1, 0, 0]) # テストデータ


alg = NaiveBayesClassifier()
alg.fit(X, y)
print(alg.predict(t))
print(alg.sorted_important_features_index())
