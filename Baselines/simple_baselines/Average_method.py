#baseline
#For each disease, take the temporal or spatial average
#as a complementary value.
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
diease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']

def get_tensor_A(rate, yy, select_diease_id):
    year = yy
    diease_select = [v for (i, v) in enumerate(diease_list) if i in select_diease_id]
    diease_name = diease_select[0]
    #print("disease：", diease_name)
    N1 = 483
    N2 = len(list(select_diease_id))
    N3 = 2017 - year + 1

    R = np.ones((N3, N1, N2), dtype='float64')
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
    ward_number = int(N1 * (100 - rate) / 100)  #

    for y in range(N3 - 1, N3):
        # print(y)
        data_year = R[y]

        ward_list = []  #
        ward_nor_list = []  #
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

    return R, ward_nor_list, ward_list

def test_loss(R_result, ward_nor_list, yy, diease_id, dimention):

    print(""+diease_list[diease_id]+" results：")
    #print(R_result)
    year=yy

    df_diease = pd.read_csv("../data/Chronic_Diseases_Prevalence_Dataset.csv")
    n=0
    y=0
    y_mae=0

    aaa = df_diease[diease_list[diease_id]+"_2017"]
    R_original = np.ones((1,dimention), dtype='float64')
    R_original=aaa.values

    for id in ward_nor_list:
        result=R_result[id]
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

def Spatial_average(select_diease_id,missing_rate):
    print("Spatial_average")
    for yy in range(2008,2009):
        seed = 225
        random.seed(seed)
        np.random.seed(seed)

        A, ward_nor_list, ward_list = get_tensor_A(missing_rate, yy, select_diease_id)
        ave_all = [A[-1,i,0] for i in ward_list]

        average = sum(ave_all)/len(ave_all)

        A_result = [average for i in range(483)]
        RMSE, MAE = test_loss(A_result, ward_nor_list, yy, select_diease_id[0], 483)


def Time_average(select_diease_id,missing_rate):
    print("Time_average")
    for yy in range(2008,2009):
        seed = 225
        random.seed(seed)
        np.random.seed(seed)
        A, ward_nor_list, ward_list = get_tensor_A(missing_rate, yy, select_diease_id)
        A_result = [0 for i in range(483)]
        for i in ward_nor_list:
            time_list = A[:, i, 0]
            time_list = list(time_list[0:len(time_list)-1])
            #print(time_list)
            while 0 in time_list:
                time_list.remove(0)
            if len(time_list)==0:
                #print("cont")
                continue
            average = sum(time_list) / len(time_list)
            A_result[i] = average

        #sys.exit(1)
        RMSE, MAE = test_loss(A_result, ward_nor_list, yy, select_diease_id[0], 483)


if __name__=='__main__':

    select_diease_id = [0]
    missing_rate = 50
    print("target year 2017, missing rate =", str(missing_rate))
    Spatial_average(select_diease_id,missing_rate)
    print()
    Time_average(select_diease_id,missing_rate)