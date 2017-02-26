from flask import Blueprint, request, jsonify, json
from sqwak.models import db, MlClass, MlApp, AudioSample
from sqwak.schemas import ma, ml_class_schema, ml_classes_schema
from sqlalchemy import text


premade_ml_class_controller = Blueprint('premade_ml_class', __name__)


@premade_ml_class_controller.route("", methods=['GET'])
def ml_class_collection():
  # premade_ml_classes = MlClass.query.filter_by(ml_app_id=None)
  sql = """SELECT ml_class.*, COUNT(audio_sample.id) AS num_samples FROM ml_class, audio_sample 
      WHERE ml_app_id IS NULL AND ml_class.id = audio_sample.ml_class_id
      GROUP BY ml_class.id;
    """
  result = db.engine.execute(text(sql))
  premade_classes = []
  for row in result:
    premade_classes.append(dict(row))

  return jsonify({
    "data": premade_classes
  })


@premade_ml_class_controller.route("/<int:class_id>/copy", methods=['POST'])
def copy(class_id):
  premade_ml_class = MlClass.query.filter_by(id=class_id, ml_app_id=None).first_or_404()
  
  ml_app_id = request.json['to_app_id']
  ml_app = MlApp.query.filter_by(id=ml_app_id).first_or_404()

  # Copy the class
  ml_class = MlClass(
      ml_app_id=ml_app_id,
      class_name=premade_ml_class.class_name,
      img_name=premade_ml_class.img_name,
      package_name=str(ml_app_id) + "." + premade_ml_class.class_name)
  db.session.add(ml_class)
  db.session.flush()
  db.session.commit()

  sql = """INSERT INTO audio_sample( 
      label, features, extraction_method, in_point, out_point, salience, ml_class_id
    )
    SELECT label, features, extraction_method, in_point, out_point, salience, {new_class_id} 
    FROM audio_sample 
    WHERE ml_class_id={original_class_id}""".format(new_class_id=ml_class.id, original_class_id=class_id)

  result = db.engine.execute(text(sql))
  return ml_class_schema.jsonify(ml_class)