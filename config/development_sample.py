# -*- coding:utf-8 -*-
import os
from .default import Config


class DevelopmentConfig(Config):
    # App config
    DEBUG = True

    # SQLAlchemy config
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql://root:password@localhost:3306/database'
