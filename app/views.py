# -*- Encoding: utf-8 -*-
from flask import Blueprint, render_template,\
    session, redirect, url_for
from .forms import NameForm
from .models.accounts import User
from .models import db

bp = Blueprint('default', __name__)


@bp.route('/', methods=['GET', 'POST'])
def index():
    form = NameForm()
    if form.validate_on_submit():
        session['name'] = form.name.data
        return redirect(url_for('default.index'))
    return render_template('default/index.html',
                           form=form, name=session.get('name'))


@bp.route('/user/<name>')
def user(name):
    return render_template('default/user.html', name=name)


@bp.route('/test/<name>')
def test(name):
    user = User(name)
    print(user.__repr__())
    #db.session.add(user)
    #db.session.commit()
    users = User.query.all()
    print(users)
    userResult = User.query.filter_by(username=name).first()
    return render_template('default/user.html', name=userResult.username)
