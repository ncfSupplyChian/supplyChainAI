# -*- Encoding: utf-8 -*-
from flask import Blueprint, jsonify
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

bp = Blueprint('apis', __name__)


@bp.route('/api/v1/kmeans/<kvalue>', methods=['GET'])
def kmeans(kvalue):
    dataMat = []
    fr = open("E:\\code\\python\\data\\dshl_kmeans.txt")
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 映射所有的元素为 float（浮点数）类型
        dataMat.append(fltLine)
    km = KMeans(n_clusters=int(kvalue))  # 初始化
    km.fit(dataMat)  # 拟合
    km_preds = km.predict(dataMat).tolist()  # 预测
    centers = km.cluster_centers_.tolist()  # 质心
    result_list = []
    for i, centerPoint in enumerate(centers):
        result = {'kIndex': i + 1, 'centerPoint': centerPoint, 'dataCount': 0}
        result_list.append(result)
    for km_pred in km_preds:
        result_list[km_pred]['dataCount'] = result_list[km_pred]['dataCount'] + 1
    ch_value = metrics.calinski_harabaz_score(dataMat, km_preds)
    return jsonify({'resultList': result_list, 'chValue': ch_value})


@bp.route('/api/v1/logisticRegression', methods=['GET'])
def logistic_regression():
    xDataMat = []
    fr = open("E:\\code\\python\\data\\dshl_lr_X.txt")
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        xDataMat.append(fltLine)
    yDataMat = []
    fr = open("E:\\code\\python\\data\\dshl_lr_Y.txt")
    for line in fr.readlines():
        fltLine = float(line.replace('\n', ''))
        yDataMat.append(fltLine)

    lr = LogisticRegression()
    lr.fit(xDataMat, yDataMat)
    print(lr.coef_.tolist())
    print(lr.intercept_.tolist())
    #print(lr.score(xDataMat, yDataMat))
    score = 0
    for i, xData in enumerate(xDataMat):
        sig = 1 / (1 + np.exp(-(np.dot(xData, lr.coef_.tolist()[0]) + lr.intercept_)))
        # print(sig)
        # temp_x = []
        # temp_x.append(xData)
        # print(lr.predict_proba(temp_x))
        if sig > 0.5:
            result = 1
        elif sig <= 0.5:
            result = 0
        if result == yDataMat[i]:
            score += 1
    my_score = float(score) / float(len(xDataMat))
    print(my_score)
    print(lr.score(xDataMat, yDataMat))
    return "done"

