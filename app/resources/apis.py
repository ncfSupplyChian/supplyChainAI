# -*- Encoding: utf-8 -*-
from flask import Blueprint, jsonify
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

bp = Blueprint('apis', __name__)


@bp.route('/api/v1/kmeans/<kvalue>', methods=['GET'])
def kmeans(kvalue):
    dataMat = []
    fr = open("E:\\code\\python\\kmeans\\testset4.txt")
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
        result = {'kIndex': i, 'centerPoint': centerPoint, 'dataCount': 0}
        result_list.append(result)
    for km_pred in km_preds:
        result_list[km_pred]['dataCount'] = result_list[km_pred]['dataCount'] + 1
    ch_value = metrics.calinski_harabaz_score(dataMat, km_preds)
    return jsonify({'resultList': result_list, 'chValue': ch_value})

