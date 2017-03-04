from flask import Blueprint, request, jsonify, json
from werkzeug import secure_filename
from sqwak.models import db, MlClass, AudioSample
from sqwak.schemas import ma, ml_class_schema, audio_sample_schema
from sqwak.services import feature_extractor
from sqwak.services import amplitude_extractor
from slugify import slugify


audio_sample_controller = Blueprint('audio_sample', __name__)

ALLOWED_EXTENSIONS = set(['wav'])

@audio_sample_controller.route("", methods=['POST'])
def audio_sample_collection(user_id, app_id, class_id):
    ml_class = MlClass.query.filter_by(id=class_id, ml_app_id=app_id).first_or_404()
    file = request.files['file']

    amps, sample_rate = amplitude_extractor.extract(file)
    features = feature_extractor.extract(amps, sample_rate)
    
    audio_sample = AudioSample(
        ml_class_id=ml_class.id,
        features=features,
        extraction_method="urban_sound_1"
    )

    db.session.add(audio_sample)
    db.session.commit()

    return ml_class_schema.jsonify(ml_class)