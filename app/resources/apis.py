# -*- Encoding: utf-8 -*-
from flask import Blueprint, jsonify
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

bp = Blueprint('api', __name__)

tasks =[
    {
        'id':1,
        'title': 'Buy groceries',
        'description': 'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': 'Learn Python',
        'description': 'Need to find a good Python tutorial on the web',
        'done': False
    }]


@bp.route('/api/v1/test', methods=['GET'])
def test():
    dataMat = []
    fr = open("E:\\code\\python\\kmeans\\testset4.txt")  # 注意，这个是相对路径，请保证是在 MachineLearning 这个目录下执行。
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 映射所有的元素为 float（浮点数）类型
        dataMat.append(fltLine)
    km = KMeans(n_clusters=24)  # 初始化
    km.fit(dataMat)  # 拟合
    km_pred = km.predict(dataMat)  # 预测
    centers = km.cluster_centers_  # 质心
    # for temp in centers:
    #     for tempPoint in temp:
    #         print(tempPoint)
    #     print("")
    f = open("E:\\code\\python\\kmeans\\result.txt", "w")
    for test in km_pred:
        print(test, file=f)
    f.close()
    return jsonify({'tasks': tasks})

