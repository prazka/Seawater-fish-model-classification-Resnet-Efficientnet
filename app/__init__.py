from flask import Flask
from .routes.predict import predict_route
from .config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.register_blueprint(predict_route)
    return app
