from flask import Blueprint, request, jsonify
from sqwak.models import db, MlApp, User
from sqwak.schemas import ml_app_schema, ml_classes_schema
from sqwak.forms.MlApp import NewMlAppForm
from sqwak.errors import InvalidUsage
from sqwak.services import model_manager
from sqwak.services import feature_extractor
from sqwak.services import amplitude_extractor
import numpy as np
from sqlalchemy.orm import joinedload


ml_app_controller = Blueprint('ml_app', __name__)


@ml_app_controller.route("", methods=['GET', 'POST'])
def all_apps(user_id):
  form = NewMlAppForm(request.form)
  if request.method == 'POST' and form.validate():
    # CREATE THE APP IN THE DB
    user = User.query.filter_by(id=user_id).first_or_404()
    ml_app = MlApp(app_name=form.app_name.data, owner_id=user_id)
    db.session.add(ml_app)
    db.session.commit()
    return ml_app_schema.jsonify(ml_app)

  elif list(form.errors.items()):
    for fieldName, errorMessages in form.errors.items():
      for err in errorMessages:
        raise InvalidUsage(err, status_code=400)
  else:
    ml_apps = MlApp.query.filter_by(owner_id=user_id).options(
        joinedload(MlApp.ml_classes)).all()

    res = []
    for ml_app in ml_apps:
      num_samples = ml_app.num_samples
      raw_ml_app = ml_app_schema.dump(ml_app)
      raw_ml_app['num_samples'] = num_samples
      raw_ml_app['ml_classes'] = ml_classes_schema.dump(ml_app.ml_classes)
      res.append(raw_ml_app)
    return jsonify(res)


@ml_app_controller.route("/<int:app_id>", methods=['GET', 'DELETE'])
def one_app(user_id, app_id):
  if request.method == 'GET':
    ml_app = MlApp.query.filter_by(owner_id=user_id, id=app_id).options(
        joinedload(MlApp.ml_classes)).first_or_404()
    ml_classes = []
    for ml_class in ml_app.ml_classes:
      num_samples = ml_class.audio_samples.count()
      ml_class_dict = ml_classes_schema.dump(ml_class, many=False)
      ml_class_dict['num_samples'] = num_samples
      ml_classes.append(ml_class_dict)
    res = ml_app_schema.dump(ml_app)
    res['ml_classes'] = ml_classes

    return jsonify(res)
  else:
    ml_app = MlApp.query.filter_by(owner_id=user_id, id=app_id).first_or_404()
    db.session.delete(ml_app)
    db.session.commit()
    return jsonify({"status_code": 204})


@ml_app_controller.route("/<int:app_id>/train", methods=['POST'])
def train(user_id, app_id):
  ml_app = MlApp.query.filter_by(owner_id=user_id, id=app_id).first_or_404()

  training_data = ml_app.training_data
  training_data = np.array(training_data, dtype=object)

  pickled_model = model_manager.create_model(training_data)
  ml_app.working_model = pickled_model
  ml_app.working_model_dirty = False
  db.session.commit()

  res = ml_app_schema.dump(ml_app)
  return jsonify(res)


@ml_app_controller.route("/<int:app_id>/test", methods=['POST'])
def test(user_id, app_id):
  ml_app = MlApp.query.filter_by(owner_id=user_id, id=app_id).first_or_404()
  file = request.files['file']

  amps, sample_rate = amplitude_extractor.extract(file)
  features = feature_extractor.extract(amps, sample_rate)
  predictions = model_manager.predict(ml_app.working_model, features)

  return {
      "dog": 67.89,
      "cat": 32.11
  }

  return jsonify(predictions)


@ml_app_controller.route("/<int:app_id>/predict", methods=['POST'])
def predict(user_id, app_id):
  ml_app = MlApp.query.filter_by(owner_id=user_id, id=app_id).first_or_404()
  file = request.files['file']

  amps, sample_rate = amplitude_extractor.extract(file)
  features = feature_extractor.extract(amps, sample_rate)

  predictions = {
      "error": "app is not published"
  }
  if ml_app.published_model:
    predictions = model_manager.predict(ml_app.working_model, features)

  return jsonify(predictions)


@ml_app_controller.route("/<int:app_id>/publish", methods=['POST'])
def publish(user_id, app_id):
  ml_app = MlApp.query.filter_by(owner_id=user_id, id=app_id).first_or_404()
  if (ml_app.working_model):
    ml_app.published_model = ml_app.working_model
    ml_app.last_published = 'now'
    db.session.commit()

  res = ml_app_schema.jsonify(ml_app)
  return res
