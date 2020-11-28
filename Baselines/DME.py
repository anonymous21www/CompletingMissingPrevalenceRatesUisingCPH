#DME: The Deep Multimodal Encoding
import tensorflow as tf
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
diease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']

def test_loss(R_result, ward_nor_list, yy, diease_id, dimention):
    print(str(yy)+""+diease_list[diease_id]+"results：")
    #print(R_result)
    year=yy
    df_diease = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    n=0
    y=0
    y_mae=0
    y_per=0
    aaa = df_diease[diease_list[diease_id]+"_2017"]
    R_original = np.ones((1,dimention), dtype='float64')
    R_original=aaa.values
    for id in ward_nor_list:
        result=R_result[id]
        origial=R_original[id]
        if str(origial)!="nan" and origial!=0:
            #print(origial[rate], result[rate])
            y=y+pow((origial-result),2)
            n+=1
            y_mae = y_mae + abs(origial - result)
            y_per = y_per + (abs(origial - result) / abs(origial))
    #print(y,n)
    RMSE=sqrt(y/n)
    MAE=y_mae/n
    PER = y_per / n
    print("RMSE:",RMSE)
    print("MAE:",MAE)
    return RMSE, MAE

def get_tensor_A(rate, yy, select_diease_id):
        year = yy  #
        diease_select=[v for (i,v) in enumerate(diease_list) if i in select_diease_id]
        print("disease：",diease_select)
        N1 = 483
        N2 = len(list(select_diease_id))
        N3 = year - 2008 + 1

        R = np.ones((N3, N1, N2), dtype='float64')  # ?x624x18
        ward_code_list=[]

        modality5 = np.ones((N3, N1, 1), dtype='float64')  #
        modality6 = np.ones((N3, N1, 1), dtype='float64')  #

        diease_select_list = 0
        diease_select_list2 = 1  #
        diease_select_list3 = 2  #

        for y in range(2008,year+1):
            df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
            ward_code_list = list(df['Ward Code'])
            # print(list(df))
            df1 = df[diease_list[diease_select_list] + "_" + str(y)]
            R[y-2008,:,0] = df1.values

            df_d1 = df[diease_list[diease_select_list2] + "_" + str(y)]
            modality5[y - 2008] = np.array(df_d1.values).reshape((N1, 1))
            df_d2 = df[diease_list[diease_select_list3] + "_" + str(y)]
            modality6[y - 2008] = np.array(df_d2.values).reshape((N1, 1))

            for tt in range(N1):
                if np.isnan(modality5[y - 2008,tt,0]) or np.isnan(modality6[y - 2008,tt,0]):
                    print("error")
                    sys.exit(1)

        for i in range(0, len(R)):
            for j in range(0, len(R[0])):
                for k in range(0, len(R[0][0])):
                    if np.isnan(R[i][j][k]):
                        R[i][j][k] = 0
        R_original = R
        ward_number = int(N1 * (100 - rate) / 100)  #
        for y in range(N3 - 1, N3):
            # print(y)
            data_year = R[y][:][:]
            ward_list = []  #
            ward_nor_list = []  #
            num = 0
            df_ward = pd.read_csv("./data/Variance_2008_2017_" + diease_list[diease_select_list] + "_NORMALIZE.csv")
            df_diease = pd.read_csv("./data/Ward_code_list.csv")
            ward_code_old=list(df_diease['Ward Code'])

            ward_var = list(df_ward["Ward_id_" + str(2008) + "_" + str(year)])
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

        for y in range(N3 - 1, N3):
            data_year = modality5[y][:][:]
            ward_list2 = []  #
            ward_nor_list2 = []  #
            num = 0
            df_ward = pd.read_csv("./data/Variance_2008_2017_" + diease_list[diease_select_list2] + "_NORMALIZE.csv")
            df_diease = pd.read_csv("./data/Ward_code_list.csv")
            ward_code_old=list(df_diease['Ward Code'])

            ward_var = list(df_ward["Ward_id_" + str(2008) + "_" + str(year)])
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
                        ward_list2.append(index1)
                    # print(num)

            for i in range(N1):
                if i in ward_list2:
                    continue
                ward_nor_list2.append(i)
                modality5[y][i][:] = [0] * N2

        for y in range(N3 - 1, N3):
            data_year = modality6[y][:][:]
            ward_list3 = []  #
            ward_nor_list3 = []  #
            num = 0
            df_ward = pd.read_csv("./data/Variance_2008_2017_" + diease_list[diease_select_list3] + "_NORMALIZE.csv")
            df_diease = pd.read_csv("./data/Ward_code_list.csv")
            ward_code_old=list(df_diease['Ward Code'])

            ward_var = list(df_ward["Ward_id_" + str(2008) + "_" + str(year)])
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
                        ward_list3.append(index1)
                    # print(num)

            for i in range(N1):
                if i in ward_list3:
                    continue
                ward_nor_list3.append(i)
                modality6[y][i][:] = [0] * N2

        return R, modality5, modality6, ward_nor_list

if __name__=='__main__':
    select_diease_id = 0
    df = pd.DataFrame()
    year = 2017
    missing_rate = 90
    print("target year 2017, missing rate =", str(missing_rate))
    for diease_select in range(select_diease_id, select_diease_id + 1):
        select_diease_id = [diease_select]
        for yy in range(2008, 2009):
            seed = 225
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)
            A,modality_diease1,modality_diease2,ward_nor_list=get_tensor_A(missing_rate,year,select_diease_id)
            input_dimension=len(A[0])
            #sys.exit(0)
            id_0 = []
            id_4 = []
            id_5 = []

            input_x=np.array([[i for i in range(input_dimension*3)]])
            for j in range(len(A)-1):
                modality5_sample = modality_diease1[j]
                modality5_sample = np.array(modality5_sample).reshape((1, input_dimension))

                modality6_sample = modality_diease2[j]
                modality6_sample = np.array(modality6_sample).reshape((1, input_dimension))

                for k in range(len(select_diease_id)):
                    #print()
                    A1=A[j,:,k]
                    A2=np.array(A1).reshape((1,input_dimension))
                    data = np.concatenate([A2, modality5_sample, modality6_sample], axis=1)
                    input_x=np.vstack((input_x,data))
                    id=np.where((A2)!=0)
                    #print(966-len(np.where((A2)==0)[0]))
                    for i in range(len(id[0])):
                        temp=[]
                        temp.append(list(id[0])[i]+j)
                        temp.append(list(id[1])[i])
                        id_0.append(temp)

                    id=np.where((modality5_sample)!=0)
                    for i in range(len(id[0])):
                        temp=[]
                        temp.append(list(id[0])[i]+j)
                        temp.append(list(id[1])[i])
                        id_4.append(temp)

                    id=np.where((modality6_sample)!=0)
                    for i in range(len(id[0])):
                        temp=[]
                        temp.append(list(id[0])[i]+j)
                        temp.append(list(id[1])[i])
                        id_5.append(temp)

            input_x=input_x[1:,:]

            input_pre=np.array([[i for i in range(input_dimension*3)]])
            modality5_sample = modality_diease1[-1]
            modality5_sample = np.array(modality5_sample).reshape((1, input_dimension))
            modality6_sample = modality_diease2[-1]
            modality6_sample = np.array(modality6_sample).reshape((1, input_dimension))
            for k in range(len(select_diease_id)):
                # print()
                A1 = A[len(A)-1, :, k]
                A2 = np.array(A1).reshape((1, input_dimension))
                data = np.concatenate([A2, modality5_sample, modality6_sample], axis=1)
                input_pre = np.vstack((input_pre, data))
            input_pre = input_pre[1:, :]

            # Parameter
            learning_rate = 0.0004
            training_epochs = 5 #
            batch_size = 256
            display_step = 1
            examples_to_show = 1
            weight_decay = 0.5
            times = 100000
            # Network Parameters
            n_input = input_dimension*3  # MNIST data input (img shape: 28*28)
            dim_x=624
            dim_y=18

            X=tf.placeholder("float",[None,n_input])

            # hidden layer settings
            #32 32
            n_hidden_1 = 32 # 1st layer num features
            n_hidden_2 = 32 # 2nd layer num features
            weights = {
                'encoder_h11':tf.Variable(tf.random_normal([input_dimension,n_hidden_1], mean=0.05, stddev=0.05, seed=225 )),
                'encoder_h12': tf.Variable(tf.random_normal([input_dimension, n_hidden_1], mean=0.05, stddev=0.05, seed=225)),
                'encoder_h13': tf.Variable(
                    tf.random_normal([input_dimension, n_hidden_1], mean=0.05, stddev=0.05, seed=225)),

                'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1*3,n_hidden_2], mean=0.05, stddev=0.05, seed=225)),
                'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1*3], mean=0.05, stddev=0.05, seed=225)),

                'decoder_h21': tf.Variable(tf.random_normal([n_hidden_1, input_dimension], mean=0.05, stddev=0.05, seed=225)),
                'decoder_h22': tf.Variable(tf.random_normal([n_hidden_1, input_dimension], mean=0.05, stddev=0.05, seed=225)),
                'decoder_h23': tf.Variable(
                    tf.random_normal([n_hidden_1, input_dimension], mean=0.05, stddev=0.05, seed=225)),
            }
            biases = {
                'encoder_b11': tf.Variable(tf.random_normal([n_hidden_1], seed=225)),
                'encoder_b12': tf.Variable(tf.random_normal([n_hidden_1], seed=225)),
                'encoder_b13': tf.Variable(tf.random_normal([n_hidden_1], seed=225)),

                'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2], seed=225)),
                'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1*3], seed=225)),
                'decoder_b21': tf.Variable(tf.random_normal([input_dimension], seed=225)),
                'decoder_b22': tf.Variable(tf.random_normal([input_dimension], seed=225)),
                'decoder_b23': tf.Variable(tf.random_normal([input_dimension], seed=225)),

                }

            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['encoder_h11'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['encoder_h12'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['encoder_h13'])

            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['encoder_h2'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['decoder_h1'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['decoder_h21'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['decoder_h22'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['decoder_h23'])

            # Building the encoder
            def encoder(x):
                #print(x)
                result = tf.split(x, axis=1, num_or_size_splits=3)
                # Encoder Hidden layer with sigmoid activation #1
                #print(result[0])
                layer_11 = tf.nn.sigmoid(tf.add(tf.matmul(result[0], weights['encoder_h11']),
                                               biases['encoder_b11']))
                layer_12 = tf.nn.sigmoid(tf.add(tf.matmul(result[1], weights['encoder_h12']),
                                                biases['encoder_b12']))
                layer_13 = tf.nn.sigmoid(tf.add(tf.matmul(result[2], weights['encoder_h13']),
                                               biases['encoder_b13']))

                layer_1 = tf.concat([layer_11, layer_12, layer_13], 1)
                #print(layer_1)
                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                               biases['encoder_b2']))
                #print(layer_2)
                return layer_2

            # Building the decoder
            def decoder(x):
                # Encoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                               biases['decoder_b1']))
                #print(layer_1)
                # Decoder Hidden layer with sigmoid activation #2
                result = tf.split(layer_1, axis=1, num_or_size_splits=3)

                layer_21 = tf.nn.sigmoid(tf.add(tf.matmul(result[0], weights['decoder_h21']),
                                               biases['decoder_b21']))
                layer_22 = tf.nn.sigmoid(tf.add(tf.matmul(result[1], weights['decoder_h22']),
                                               biases['decoder_b22']))
                layer_23 = tf.nn.sigmoid(tf.add(tf.matmul(result[2], weights['decoder_h23']),
                                               biases['decoder_b23']))

                layer_2 = tf.concat([layer_21, layer_22, layer_23], 1)
                #print("l2", layer_2)
                return layer_2

            # Construct model
            encoder_op = encoder(X) 			# 128 Features
            decoder_op = decoder(encoder_op)	# 784 Features

            # Prediction
            y_pred = decoder_op	# After

            # Targets (Labels) are the input data.
            y_true = X		# Before

            def loss_cal(y_true,y_pred):
                result_true = tf.split(y_true, axis=1, num_or_size_splits=3)
                result_pred = tf.split(y_pred, axis=1, num_or_size_splits=3)

                y_true2_1 = tf.gather_nd(result_true[0], id_0)
                y_true2_2 = tf.gather_nd(result_true[1], id_4)
                y_true2_3 = tf.gather_nd(result_true[2], id_5)

                y_pred2_1 = tf.gather_nd(result_pred[0], id_0)
                y_pred2_2 = tf.gather_nd(result_pred[1], id_4)
                y_pred2_3 = tf.gather_nd(result_pred[2], id_5)

                temp1 = tf.pow(y_true2_1 - y_pred2_1, 2)
                temp2 = 0.2 * tf.pow(y_true2_2 - y_pred2_2, 2)
                temp3 = 0.2 * tf.pow(y_true2_3 - y_pred2_3, 2)

                res = tf.concat([temp1, temp2, temp3], 0)
                return res

            # Define loss and optimizer, minimize the squared error
            cost = tf.reduce_mean(loss_cal(y_true,y_pred))
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            cost += reg_term

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            # Launch the graph
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(times):

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={X: input_x	})

                    if i%1000==0:
                        print("Epoch:", '%04d' % (i+1),
                              "cost=", "{:.9f}".format(c))

                print("Optimization Finished!")

                # # Applying encode and decode over test set
                A_result = sess.run(
                    y_pred, feed_dict={X: input_pre})

                for dd in range(len(select_diease_id)):
                    diease_id=select_diease_id[dd]
                    RMSE, MAE=test_loss(A_result[dd,:], ward_nor_list, year, diease_id, input_dimension)
