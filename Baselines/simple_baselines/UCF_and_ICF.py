#baseline
#CF: For each disease, User-based collaborative filtering (UCF)
#and Item-based collaborative filtering (ICF) are applied to generate
#a prediction, respectively, and the final result is the average of the
#two predictions
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
        #print("ward_list", sorted(ward_list))
        #print("len ward_list", len(ward_list))
        #print("len ward_nor_list", len(ward_nor_list))
    return R, ward_nor_list, ward_list

def test_loss(R_result, ward_nor_list, yy, diease_id, dimention):

    year=yy
    df_diease = pd.read_csv("../data/Chronic_Diseases_Prevalence_Dataset.csv")
    n=0
    y=0
    y_mae=0
    aaa = df_diease[diease_list[diease_id]+"_2017"]
    R_original = np.ones((1,dimention), dtype='float64')
    R_original=aaa.values

    for id in ward_nor_list:
        result=R_result[id][-1]
        origial=R_original[id]

        if str(origial)!="nan" and origial!=0 and result!=0:
            #print(origial, result)
            y=y+pow((origial-result),2)
            n+=1
            y_mae = y_mae + abs(origial - result)

    RMSE=sqrt(y/n)
    MAE=y_mae/n
    #print("RMSE:",RMSE)
    #print("MAE:",MAE)

    #print()
    return RMSE, MAE

def UCF(R,ward_list):
    #print(len(ward_list))
    time_slot = len(R[0])

    ward_list.sort()
    # print(ward_list)
    for i in range(0,len(R)):
        #print(len(ward_list)/2)
        id1 = random.randint(0, int(len(ward_list)/2))

        id2 = random.randint(int(len(ward_list)/2)+1, len(ward_list) - 1)
        if i in ward_list:
            continue
        #print(i)
        miss_row=R[i]

        user1=R[ward_list[id1]]
        user2=R[ward_list[id2]]

        n1=0
        n2=0
        sim1=0
        sim2=0
        for j in range(time_slot-1):
            if user1[j]!=0 and miss_row[j]!=0:
                n1+=1
                sim1=pow(user1[j]-miss_row[j],2)
            if user2[j] != 0 and miss_row[j] != 0:
                n2 += 1
                sim2 = pow(user2[j] - miss_row[j], 2)
        if sim1==0 or sim2==0:
            continue
        sim1=1/(sqrt(sim1/n1))
        sim2 = 1 / (sqrt(sim2 / n2))
        R[i][time_slot-1]=(user1[time_slot-1]*sim1+user2[time_slot-1]*sim2)/(sim1+sim2)
    return R

def UCF_method(select_diease_id,missing_rate):
    for yy in range(2008,2009):
        seed = 225
        random.seed(seed)
        np.random.seed(seed)
        A, ward_nor_list, ward_list = get_tensor_A(missing_rate, yy, select_diease_id)
        A1 = A[:,:,0]
        R_MF = UCF(A1.T, ward_list)
        RMSE, MAE = test_loss(R_MF, ward_nor_list, yy, select_diease_id[0], 483)
        return RMSE, MAE

def ICF(R,ward_list):
    #print(len(ward_list))
    time_slot = len(R[0])

    ward_list.sort()
    # print(ward_list)
    for i in range(0,len(R)):
        #print(len(ward_list)/2)
        id1 = random.randint(0, int(time_slot)-2)
        #print(id1)

        #print(i)
        miss_col=R[:,-1]
        #print(miss_col)

        item1=R[:,id1]
        #print(item1)


        n1=0
        sim1=0
        for j in range(len(R)):
            if item1[j]!=0 and miss_col[j]!=0:
                n1+=1
                sim1=pow(item1[j]-miss_col[j],2)
        if sim1==0:
            continue
        sim1=1/(sqrt(sim1/n1))

        R[i][time_slot-1]=(item1[time_slot-1]*sim1)/(sim1)
    return R

def ICF_method(select_diease_id,missing_rate):
    for yy in range(2008,2009):
        seed = 225
        random.seed(seed)
        np.random.seed(seed)

        A, ward_nor_list, ward_list = get_tensor_A(missing_rate, yy, select_diease_id)
        A1 = A[:,:,0]
        R_MF = ICF(A1.T, ward_list)

        RMSE, MAE = test_loss(R_MF, ward_nor_list, yy, select_diease_id[0], 483)
        return RMSE, MAE

if __name__=='__main__':
    select_diease_id = [0]
    print("" + diease_list[select_diease_id[0]] + " results：")
    missing_rate = 50
    print("target year 2017, missing rate =",str(missing_rate))
    RMSE, MAE = UCF_method(select_diease_id, missing_rate)
    RMSE2, MAE2 = ICF_method(select_diease_id, missing_rate)
    print("RMSE:",(RMSE + RMSE2) / 2)
    print("MAE:", (MAE + MAE2) / 2)