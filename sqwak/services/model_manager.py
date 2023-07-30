import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from math import floor
import pickle
from .models import model_1

def create_model(ml_classes, pickled=True):
  model_paramaters = model_1.train(ml_classes)
  return pickle.dumps(model_paramaters)


def predict(working_model, features):
  model_paramaters = pickle.loads(working_model)
  results = model_1.predict(model_paramaters, features)

  return results

  features = np.array(features)
  features = features.reshape(1,-1) 

  predictions = clf.predict(features)
  probabilities = clf.predict_proba(features)
  probabilities = np.multiply(probabilities, 100)

  results = []

  if len(clf.classes_) <= 1:
    return [{
      'label': clf.classes_[0],
      'probability': probabilities[0][0]
    }]

  for p in probabilities:
      for i, value in enumerate(p):
          print((clf.classes_))
          result = {}
          result['label'] = clf.classes_[i]
          result['probability'] = value
          results.append(result)
  

  return results