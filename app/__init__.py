import logging
from flask import Flask
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def create_app(config_override: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="../templates")
    app.config.from_object(Config)
    if config_override:
        app.config.update(config_override)

    try:
        from app.db import init_db
        init_db()
    except Exception as exc:
        logging.getLogger(__name__).warning("DB init skipped: %s", exc)

    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    return app
