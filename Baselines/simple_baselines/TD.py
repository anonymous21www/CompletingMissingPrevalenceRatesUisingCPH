#baseline
#TD: Construct a Tensor with three dimensions (year, grid, and
#disease), and use tensor decomposition to predict the missing values
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorly import tucker_to_tensor
from time import *
import random

diease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']
def Random_gradient_descent(A,S,R,C,T,epsilon,lambda3):
    #print()
    loss_tensor = tucker_to_tensor(S, [R, C, T])
    position_nor_0 = np.where((A) != 0)
    original = A[position_nor_0]
    result = loss_tensor[position_nor_0]

    loss_t1 = sum(list(map(lambda x: abs(x[0] - x[1]) / x[0], zip(original, result))))
    loss_t1 = loss_t1 / len(original)

    #loss_t1 = sum(list(map(lambda x: pow((x[0] - x[1]), 2), zip(original, result))))
    #loss_t1 = sqrt(loss_t1)
    loss_t=loss_t1+epsilon-1
    print("loss:",loss_t1)
    #step size,1/sqrt(t++)
    t0=10
    t=t0;
    flag=0
    times = 0
    while abs(loss_t-loss_t1)>epsilon:
        if loss_t > loss_t1:
            flag = 1
        # if flag == 1 and loss_t < loss_t1:
        #     break
        for i in range(len(A)):
            for j in range(len(A[0])):
                for k in range(len(A[0][0])):
                    if A[i][j][k]==0:
                        continue
                    nita=1/sqrt(t)
                    t+=1
                    eijk = A[i][j][k] - tucker_to_tensor(S, [R[i, :], C[j, :], T[k, :]])[0]

                    RLfy=tucker_to_tensor(S,[R[i, :], C[j, :], T[k, :]],skip_factor=0)
                    CLfy=tucker_to_tensor(S,[R[i, :], C[j, :], T[k, :]],skip_factor=1)
                    TLfy=tucker_to_tensor(S,[R[i, :], C[j, :], T[k, :]],skip_factor=2)
                    SLfy=np.ones((len(S),len(S[0]),len(S[0][0])),dtype='float32')
                    temp = np.array([C[j, :]]).T * [T[k, :]]
                    for tt in range(len(S)):
                        SLfy[tt] = R[i, tt] * temp
                    #print(RLfy,CLfy,TLfy,SLfy)
                    #print("--")
                    #print(nita,lambda3,eijk)
                    R[i, :]=(1-nita*lambda3)*R[i, :]+nita*eijk*RLfy
                    C[j, :]=(1-nita*lambda3)*C[j, :]+nita*eijk*CLfy
                    T[k, :]=(1-nita*lambda3)*T[k, :]+nita*eijk*TLfy
                    S=(1-nita*lambda3)*S+nita*eijk*SLfy

        #compute function loss
        loss_tensor=tucker_to_tensor(S, [R,C,T])
        position_nor_0=np.where((A)!=0)
        original=A[position_nor_0]
        result=loss_tensor[position_nor_0]

        loss_t=loss_t1
        loss_t1=sum(list(map(lambda x: abs(x[0]-x[1])/x[0], zip(original, result))))
        loss_t1=loss_t1/len(original)

        times += 1
    print("test loss:", loss_t1)
    return S,R,C,T

def get_tensor_A(rate,yy):
    year=yy
    df_diease = pd.read_csv("../data/Chronic_Diseases_Prevalence_Dataset.csv")
    diease_list2 = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']
    N1 = len(df_diease)
    N2 = len(list(diease_list2))
    N3 = 2017 - year + 1

    ward_code_list = list(df_diease['Ward Code'])
    R = np.ones((N3, N1, N2), dtype='float64')

    for i in range(0, N3):
        aaa = df_diease.iloc[:, 1 + (i) * 18:19 + (i) * 18]
        R[:][:][i] = aaa.values

    for i in range(0,len(R)):
        for j in range(0,len(R[0])):
            for k in range(0,len(R[0][0])):
                if np.isnan(R[i][j][k]):
                    R[i][j][k]=0
    R_original=R

    ward_number = int(N1*(100-rate)/100)

    for y in range(N3-1,N3):
        data_year = R[y][:][:]

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
                # print(num)

        for i in range(N1):
            if i in ward_list:
                continue
            ward_nor_list.append(i)
            R[y][i][:] = [0] * N2
        print("ward_list", sorted(ward_list))
        print("len ward_list", len(ward_list))
        print("len ward_nor_list", len(ward_nor_list))

    return R,ward_nor_list

def test_loss(R_result,ward_nor_list,yy,select_diease_id):
    year=yy
    n=0
    y=0
    y_mae=0
    y_per=0

    df_diease = pd.read_csv("../data/Chronic_Diseases_Prevalence_Dataset.csv")
    i=2017-2008
    aaa = df_diease.iloc[:,  1 + (i) * 18:19 + (i) * 18]
    R_original = np.ones((426,18), dtype='float64')
    R_original=aaa.values
    diease_list2 = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                    'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                    'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                    'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                    'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                    'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                    'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']
    id11 = diease_list2.index(diease_list[select_diease_id[0]])
    for id in ward_nor_list:
        result=R_result[id]
        origial=R_original[id]

        rate = id11
        if str(origial[rate])!="nan" and origial[rate]!=0:
            y=y+pow((origial[rate]-result[rate]),2)
            n+=1
            y_mae = y_mae + abs(origial[rate] - result[rate])

    RMSE=sqrt(y/n)
    MAE=y_mae/n
    print("RMSE:",RMSE)
    print("MAE:",MAE)
    return RMSE, MAE

def TD(rate,core_dimension_1,core_dimension_2,core_dimension_3,ee,l3,yy,select_diease_id):
    A, ward_nor_list = get_tensor_A(rate,yy)
    dim1 = len(A)
    dim2 = len(A[0])
    dim3 = len(A[0][0])

    # size of core Tensor
    dimX = core_dimension_1
    dimY = core_dimension_2
    dimZ = core_dimension_3
    S = np.random.uniform(0, 0.1, (dimX, dimY, dimZ))
    R = np.random.uniform(0, 0.1, (dim1, dimX))
    C = np.random.uniform(0, 0.1, (dim2, dimY))
    T = np.random.uniform(0, 0.1, (dim3, dimZ))
    #print(R,C,T)
    nS, nR, nC, nT = Random_gradient_descent(A, S, R, C, T,ee,l3)
    A_result = tucker_to_tensor(nS, [nR, nC, nT])

    rmse, mae = test_loss(A_result[dim1 - 1][:][:], ward_nor_list,yy,select_diease_id)  # 计算测试误差

    return rmse,mae

if __name__=='__main__':

    select_diease_id = [0]
    missing_rate = 50
    print("target year 2017, missing rate =", str(missing_rate))
    seed=225
    for year in range(2008,2009):
        data_rate=[missing_rate]
        core_dimension_1 = [10]
        core_dimension_2 = [10]
        core_dimension_3 = [10]
        #0.00006 0.0001
        epsilon_list=[0.0003]
        #0.0005
        lambda3_list=[0.0005]
        #print(data_rate)
        for rate in data_rate:
            for core in range(len(core_dimension_1)):
                for ee in epsilon_list:
                    for l3 in lambda3_list:
                        random.seed(seed)
                        np.random.seed(seed)
                        rmse, mae=TD(rate,core_dimension_1[core],core_dimension_2[core],core_dimension_3[core],ee,l3,year,select_diease_id)






