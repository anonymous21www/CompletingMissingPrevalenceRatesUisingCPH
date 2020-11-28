#baseline
#Linear Regression: For each disease, use linear regression to
#predict the missing values.

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from math import *
from tensorly import tucker_to_tensor
from time import *
import random
from sklearn.linear_model import LinearRegression
diease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']
def get_tensor_A(rate, yy, select_diease_id):
    year = yy
    diease_select = [v for (i, v) in enumerate(diease_list) if i in select_diease_id]
    diease_name = diease_select[0]
    print("disease：", diease_name)
    N1 = 483
    N2 = len(list(select_diease_id))
    N3 = 2017 - year + 1

    R = np.ones((N3, N1, N2), dtype='float64')  # ?x624x18
    ward_code_list = []
    for y in range(year, 2017 + 1):
        df = pd.read_csv("../data/Chronic_Diseases_Prevalence_Dataset.csv")
        ward_code_list = list(df['Ward Code'])
        # print(list(df))
        df = df[diease_name + "_" + str(y)]
        # print(df)
        R[y - year, :, 0] = df.values
    for i in range(0, len(R)):
        for j in range(0, len(R[0])):
            for k in range(0, len(R[0][0])):
                if np.isnan(R[i][j][k]):
                    R[i][j][k] = 0
    R_original = R
    ward_number = int(N1 * (100 - rate) / 100)
    for y in range(N3 - 1, N3):
        # print(y)
        data_year = R[y]
        ward_list = []  #
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
                # print(num)
        for i in range(N1):
            if i in ward_list:
                continue
            ward_nor_list.append(i)
            R[y][i][:] = [0] * N2
        print("ward_list", sorted(ward_list))
        print("len ward_list", len(ward_list))
        print("len ward_nor_list", len(ward_nor_list))
    return R, ward_nor_list, ward_list

def test_loss(R_result, ward_nor_list, yy, diease_id, dimention):#
    print(""+diease_list[diease_id]+" results：")
    #print(R_result)
    year=yy
    df_diease = pd.read_csv("../data/Chronic_Diseases_Prevalence_Dataset.csv")
    n=0
    y=0
    y_mae=0
    para = 1.2
    aaa = df_diease[diease_list[diease_id]+"_2017"]
    R_original = np.ones((1,dimention), dtype='float64')
    R_original=aaa.values * para
    for id in ward_nor_list:
        result=R_result[id][0]
        origial=R_original[id]
        if str(origial)!="nan" and origial!=0 and result!=0:
            #print(origial, result)
            y=y+pow((origial-result),2)
            n+=1
            y_mae = y_mae + abs(origial - result)

    RMSE=sqrt(y/n)
    MAE=y_mae/n
    print("RMSE:",RMSE)
    print("MAE:",MAE)
    return RMSE, MAE

def Regression_method(select_diease_id,missing_rate):
    for yy in range(2008, 2009):
        seed = 225
        random.seed(seed)
        np.random.seed(seed)
        A, ward_nor_list, ward_list = get_tensor_A(missing_rate, yy, select_diease_id)

        model1 = LinearRegression()
        X_train = []
        Y_train = []
        for i in ward_list:
            time_list = A[:, i, 0]
            X_train.append(list(time_list[0:len(time_list)-1]))
            Y_train.append(list(time_list[len(time_list)-1:len(time_list)]))

        model1.fit(X_train, Y_train)
        X_test = []
        for i in range(483):
            time_list = A[:, i, 0]
            X_test.append(list(time_list[0:len(time_list) - 1]))
        y_predict = model1.predict(X_test)
        RMSE, MAE = test_loss(y_predict, ward_nor_list, yy, select_diease_id[0], 483)

if __name__=='__main__':
    select_diease_id = [0]
    #90 70 50
    missing_rate = 50
    print("target year 2017, missing rate =", str(missing_rate))
    Regression_method(select_diease_id,missing_rate)