# supplyChainAI
flask sklearn 开发模型

```shell
$ virtualenv venv
$ source venv/bin/activate
$ sudo pip install -r requirements.txt
```
创建flask-migrate用文件夹并执行数据库更新
```shell
$ python manage.py db init
$ python manage.py db migrate -m "initial migration"
$ python manage.py db upgrade
```