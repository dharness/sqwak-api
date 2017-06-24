pip install -r requirements.txt
brew install ffmpeg 
gunicorn --config gunicorn_config.py sqwak.__init__:app