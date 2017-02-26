from flask import Blueprint, request, abort, jsonify, json
from sqlalchemy import exc
from urllib import urlencode
from sqwak.models import User, db
from sqwak.schemas import ma, user_schema, users_schema
from sqwak.forms.User import UserForm
from sqwak.errors import InvalidUsage
from sqwak.services import auth
from flask_bcrypt import generate_password_hash
import requests
import jwt

user_controller = Blueprint('user', __name__)


@user_controller.route("", methods=['GET', 'POST'])
def user():
    if request.method == 'POST':
        body = request.get_json()
        email = body['email']
        password = body['password']

        try:
            user = User(email=email, password=password)
            db.session.add(user)
            db.session.commit()
        except exc.IntegrityError as e:
            abort(409)

        jwt = auth.create_token(user)
        res = user_schema.dump(user).data
        res['token'] = jwt
        del res['_password']
        return jsonify(res)

    else:
      users = User.query.all()
      return users_schema.jsonify(users)


@user_controller.route("/login", methods=['POST'])
def login():
    body = request.get_json()
    email = body['email']
    password = body['password']
    user = User.query.filter_by(email=email).first_or_404()

    if user.is_correct_password(password):
        jwt = auth.create_token(user)
        res = user_schema.dump(user).data
        res['token'] = jwt
        del res['_password']
        return jsonify(res)
    else:
        abort(400)

@user_controller.route("/<string:user_id>", methods=['GET'])
def user_apps(user_id):
    user = User.query.get(user_id)
    if not user:
        abort(404)
    return user_schema.jsonify(user)