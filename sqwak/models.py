import datetime
from sqlalchemy import text
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects import postgresql
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
  __tablename__ = 'user'
  id = db.Column(db.Integer, primary_key=True, autoincrement=True)
  email = db.Column(db.String(64), nullable=False, unique=True)
  email_confirmed = db.Column(db.Boolean, default=False)
  _password = db.Column(db.String(128))

  ml_apps = db.relationship('MlApp', backref='user', lazy='dynamic')

  @hybrid_property
  def password(self):
    return self._password

  @password.setter
  def password(self, plaintext):
    self._password = generate_password_hash(plaintext).decode('utf-8')

  def is_correct_password(self, plaintext):
    return check_password_hash(self._password, plaintext)


class MlApp(db.Model):
  __tablename__ = 'ml_app'
  id = db.Column(db.Integer, primary_key=True, autoincrement=True)
  owner_id = db.Column(db.Integer, db.ForeignKey(
      "user.id", ondelete="CASCADE"), nullable=False)
  app_name = db.Column(db.String)
  created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
  updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
  query_url = db.Column(db.String)
  working_model = db.Column(db.String)
  working_model_dirty = db.Column(db.Boolean, default=True)
  published_model = db.Column(db.String)
  last_published = db.Column(db.DateTime)
  ml_classes = db.relationship('MlClass',
                               backref="ml_app",
                               cascade="all, delete-orphan",
                               lazy='dynamic')

  @hybrid_property
  def training_data(self):
    rows = db.session.execute(
        """ SELECT class_name, features FROM ml_class, audio_sample WHERE ml_class.id=audio_sample.ml_class_id AND ml_class.ml_app_id={ml_app_id} AND ml_class.in_model=true;""".format(ml_app_id=self.id))
    return rows.fetchall()

  @hybrid_property
  def num_samples(self):
    query = text("""SELECT COUNT(*) FROM (ml_app INNER JOIN ml_class ON (ml_app.id = ml_class.ml_app_id) INNER JOIN audio_sample ON (audio_sample.ml_class_id = ml_class.id)) WHERE ml_app.id = 2;""")
    row = db.session.execute(query)
    return row.fetchone()[0]


class MlClass(db.Model):
  __tablename__ = 'ml_class'
  id = db.Column(db.Integer, db.Sequence(
      "ml_class_id_seq", start=110), primary_key=True)
  ml_app_id = db.Column(db.Integer, db.ForeignKey(
      "ml_app.id", ondelete="CASCADE"))
  class_name = db.Column(db.String, nullable=False)
  img_name = db.Column(db.String, nullable=True)
  package_name = db.Column(db.String, nullable=False)
  is_edited = db.Column(db.Boolean, default=False)
  in_model = db.Column(db.Boolean, default=False)
  created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
  updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
  audio_samples = db.relationship('AudioSample',
                                  backref="ml_class",
                                  cascade="all, delete-orphan",
                                  lazy='dynamic')


class AudioSample(db.Model):
  __tablename__ = 'audio_sample'
  id = db.Column(db.Integer, db.Sequence(
      "audio_sample_id_seq", start=9001), primary_key=True)
  ml_class_id = db.Column(db.Integer, db.ForeignKey(
      "ml_class.id", ondelete="CASCADE"), nullable=False)
  features = db.Column(postgresql.ARRAY(db.Integer), nullable=False)
  extraction_method = db.Column(db.String, nullable=False)
  in_point = db.Column(db.Integer)
  out_point = db.Column(db.Integer)
  salience = db.Column(db.Integer)
