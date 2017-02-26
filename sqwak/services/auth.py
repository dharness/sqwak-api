from datetime import datetime, timedelta
import jwt

JWT_SECRET = 'secret'
JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_SECONDS = 20


def create_token(user):
  payload = {
      'user_id': user.id,
      'email': user.email,
      'exp': datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS)
  }

  jwt_token = jwt.encode(payload, JWT_SECRET, JWT_ALGORITHM)
  return jwt_token