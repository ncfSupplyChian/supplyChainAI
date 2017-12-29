# -*- Encoding: utf-8 -*-
from .base import db
from marshmallow_jsonapi import Schema, fields
from marshmallow import validate


class CRUD:

    def add(self, resource):
        db.session.add(resource)
        return db.session.commit()

    def update(self):
        return db.session.commit()

    def delete(self, resource):
        db.session.delete(resource)
        return db.session.commit()


class Users(db.Model, CRUD):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    creation_time = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)

    def __init__(self, email, name):
        self.email = email
        self.name = name


class UsersSchema(Schema):
    not_blank = validate.Length(min=1, error='Field cannot be blank')
    id = fields.Integer(dump_only=True)
    email = fields.Email(validate=not_blank)
    name = fields.String(validate=not_blank)

    # self links
    def get_top_level_links(self, data, many):
        if many:
            self_link = "/users/"
        else:
            self_link = "/users/{}".format(data['id'])
        return {'self': self_link}

    class Meta:
        type_ = 'users'
