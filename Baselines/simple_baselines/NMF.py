#baseline
#NMF: For each disease, use Non-negative Matrix Factorization
#to predict the missing values.

import numpy
import random
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from math import *
from tensorly import tucker_to_tensor
from time import *
import random
diease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']

def matrix_factorization(R,P,Q,K,epsilon,beta,missing_rate):
    Q=Q.T
    matrix_temp = numpy.dot(P, Q)
    position_nor_0 = numpy.where((R) != 0)
    original = R[position_nor_0]
    result = matrix_temp[position_nor_0]
    loss_t1 = sum(list(map(lambda x: abs(x[0] - x[1]) / x[0], zip(original, result))))
    # print(loss_t1)
    loss_t1 = loss_t1 / len(original)
    loss_t = loss_t1 + epsilon + 1

    t0 = 100
    t=t0;
    while abs(loss_t-loss_t1)>epsilon:
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-numpy.dot(P[i,:],Q[:,j])
                    if R[i][j]>0:
                        alpha = 1 / sqrt(t)
                        t+=1
                        P[i,:]=P[i,:]+alpha*(2*eij*Q[:,j]-beta*P[i,:])
                        Q[:,j]=Q[:,j]+alpha*(2*eij*P[i,:]-beta*Q[:,j])
        loss_t = loss_t1
        matrix_temp=numpy.dot(P,Q)
        position_nor_0 = numpy.where((R) != 0)
        original = R[position_nor_0]
        result = matrix_temp[position_nor_0]
        loss_t1 = sum(list(map(lambda x: abs(x[0] - x[1]) / x[0], zip(original, result))))
        # print(loss_t1)
        loss_t1 = loss_t1 / len(original)
    return P,Q.T

def get_matrix_R(diease_name,rate,yy,select_diease_id):
    col_list=[]
    year = yy
    for y in range(year, 2017 + 1):
        col_list.append(diease_name+"_"+str(y))

    df_diease = pd.read_csv("../data/Chronic_Diseases_Prevalence_Dataset.csv")
    ward_code_list = list(df_diease['Ward Code'])
    df_diease=df_diease[col_list]

    N1 = len(df_diease)  # 624
    N2 = 2017 - year + 1
    #print(N1, N2, N3)
    R = numpy.ones((N1, N2), dtype='float64')  # 624x10

    R = df_diease.values  # 624*18
    #print(len(R),len(R[0]))
    for i in range(0,len(R)):
        for j in range(0,len(R[0])):
            if numpy.isnan(R[i][j]):
                R[i][j]=0

    ward_number = int(N1 * (100 - rate) / 100)

    data_year = R[:][:]
    ward_list = []
    ward_nor_list = []
    num = 0

    df_ward = pd.read_csv("../data/Variance_2008_2017_NORMALIZE.csv")
    df_diease = pd.read_csv("../data/Ward_code_list.csv")
    ward_code_old = list(df_diease['Ward Code'])

    ward_var = list(df_ward["Ward_id_" + str(year) + "_" + str(2017)])
    iii = 0
    while num < ward_number:
        id = ward_var[iii]
        iii += 1
        ward_code = ward_code_old[id]
        if ward_code in ward_code_list:
            index1 = ward_code_list.index(ward_code)
            diease_rate = data_year[index1]
            # print(diease_rate)
            if 0 not in diease_rate:
                num += 1
                ward_list.append(index1)

    for i in range(N1):
        if i in ward_list:
            continue
        ward_nor_list.append(i)
        R[i][N2 - 1] = 0

    print("ward_list", sorted(ward_list))
    print("len ward_list", len(ward_list))
    print("len ward_nor_list", len(ward_nor_list))
    return R,ward_nor_list

def test_loss(R_result,ward_nor_list,diease_name,yy):
    year = yy
    col_list = []
    n=0
    y=0
    y_mae=0
    df_diease = pd.read_csv("../data/Chronic_Diseases_Prevalence_Dataset.csv")
    i=year-2011
    aaa = df_diease[diease_name + "_2017"]
    R_original = numpy.ones((483,1), dtype='float64')
    R_original=aaa.values
    for id in ward_nor_list:
        result=R_result[id][-1]
        origial=R_original[id]
        if str(origial)!="nan" and origial!=0:
            y = y + pow((origial - result), 2)
            n += 1
            y_mae = y_mae + abs(origial - result)

    RMSE=sqrt(y/n)
    MAE=y_mae/n
    print("RMSE:",RMSE)
    print("MAE:",MAE)
    return RMSE,MAE


def MF(select_diease_id,missing_rate):
    seed=225
    rate = missing_rate
    for year in range(2008,2009):
        diease_name=diease_list[select_diease_id[0]]
        print(diease_name)
        random.seed(seed)
        numpy.random.seed(seed)
        R,ward_nor_list = get_matrix_R(diease_name,rate,year,select_diease_id)

        N=len(R)    #R rows
        M=len(R[0]) #R cols
        K=5
        P=numpy.random.uniform(0, 1, (N,K)) #
        Q=numpy.random.uniform(0, 1, (M,K)) #
        epsilon = 0.001
        beta = 0.0001
        #print(P)
        nP,nQ=matrix_factorization(R,P,Q,K,epsilon,beta,rate)
        #print(R)
        R_MF=numpy.dot(nP,nQ.T)
        #print(R_MF)
        RMSE,MAE =test_loss(R_MF, ward_nor_list,diease_name,year)

if __name__=='__main__':
    select_diease_id = [0]
    missing_rate = 50
    print("target year 2017, missing rate =", str(missing_rate))
    MF(select_diease_id,missing_rate)