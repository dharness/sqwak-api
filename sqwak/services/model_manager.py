import pickle
from .models import model_1


def create_model(ml_classes, pickled=True):
  model_paramaters = model_1.train(ml_classes)
  return pickle.dumps(model_paramaters)


def predict(working_model, features):
  model_paramaters = pickle.loads(working_model)
  results = model_1.predict(model_paramaters, features)

  return results
