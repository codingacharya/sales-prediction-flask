import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'supersecretkey')
    DEBUG = False
    TESTING = False
    MODEL_PATH = os.path.join('model', 'model.pkl')
    SCALER_PATH = os.path.join('model', 'scaler.pkl')
    LOG_FILE = os.path.join('logs', 'app.log')


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False


# Select configuration (default: Development)
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}
