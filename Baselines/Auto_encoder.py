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
    print(""+diease_list[diease_id]+" results：")
    #print(R_result)
    year=yy
    df_diease = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    n=0
    y=0
    y_mae=0
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
    RMSE=sqrt(y/n)
    MAE=y_mae/n
    print("RMSE:",RMSE)
    print("MAE:",MAE)
    return RMSE, MAE

def get_tensor_A(rate, yy, select_diease_id):
        year = yy
        diease_select=[v for (i,v) in enumerate(diease_list) if i in select_diease_id]
        diease_name = diease_select[0]
        print("disease：", diease_name)
        N1 = 483
        N2 = len(list(select_diease_id))
        N3 = 2017 - year + 1

        R = np.ones((N3, N1, N2), dtype='float64')  # ?x624x18
        ward_code_list=[]

        for y in range(year,2017+1):
            df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
            #print(df)
            ward_code_list=list(df['Ward Code'])
            #print(list(df))
            df = df[diease_name+"_"+str(y)]
            #print(df)
            R[y-year,:,0] = df.values

        for i in range(0, len(R)):
            for j in range(0, len(R[0])):
                for k in range(0, len(R[0][0])):
                    if np.isnan(R[i][j][k]):
                        R[i][j][k] = 0
        R_original = R
        ward_number = int(N1 * (100 - rate) / 100)
        for y in range(N3 - 1, N3):
            data_year = R[y]
            ward_list = []  #
            ward_nor_list = []
            num = 0
            df_ward = pd.read_csv("./data/Variance_2008_2017_NORMALIZE.csv")
            df_diease = pd.read_csv("./data/Ward_code_list.csv")
            ward_code_old=list(df_diease['Ward Code'])
            ward_var = list(df_ward["Ward_id_" + str(year) + "_" + str(2017)])
            iii = 0
            while num < ward_number:
                id = ward_var[iii]
                iii += 1
                # id = random.randint(0, N1 - 1)
                # if id in ward_list:
                #     continue
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
        return R, ward_nor_list

if __name__=='__main__':
    missing_rate = 50
    diease_select_list = [0]
    print("target year 2017, missing rate =", str(missing_rate))
    for diease_select in diease_select_list:
        select_diease_id = [diease_select]
        res_perl=[]
        mase_all = []
        mae_all = []
        perl_all = []
        for yy in range(2008,2009):
            seed = 225
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)
            A,ward_nor_list=get_tensor_A(missing_rate,yy,select_diease_id)
            input_dimension=len(A[0])
            #sys.exit(0)
            id_0=[]
            input_x=np.array([[i for i in range(input_dimension)]])
            for j in range(len(A)):
                for k in range(len(select_diease_id)):
                    #print()
                    A1=A[j,:,k]
                    A2=np.array(A1).reshape((1,input_dimension))
                    input_x=np.vstack((input_x,A2))
                    id=np.where((A2)!=0)
                    for i in range(len(id[0])):
                        temp=[]
                        temp.append(list(id[0])[i]+j)
                        temp.append(list(id[1])[i])
                        id_0.append(temp)
            input_x=input_x[1:,:]
            input_pre=np.array([[i for i in range(input_dimension)]])
            for k in range(len(select_diease_id)):
                # print()
                A1 = A[len(A)-1, :, k]
                A2 = np.array(A1).reshape((1, input_dimension))
                input_pre = np.vstack((input_pre, A2))
            input_pre = input_pre[1:, :]

            # Parameter
            learning_rate = 0.0005
            training_epochs = 5 #
            batch_size = 256
            display_step = 1
            times = 100000
            # Network Parameters
            n_input = input_dimension  # MNIST data input (img shape: 28*28)
            dim_x=32
            dim_y=32

            X=tf.placeholder("float",[None,n_input])
            #X=tf.placeholder("float",[None,n_input])

            # hidden layer settings
            n_hidden_1 = 32 # 1st layer num features
            n_hidden_2 = 32 # 2nd layer num features
            weights = {
                'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=225)),
                'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],  seed=225)),
                'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1],seed=225)),
                'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input], seed=225)),
            }
            biases = {
                'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1], seed=225)),
                'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2], seed=225)),
                'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1], seed=225)),
                'decoder_b2': tf.Variable(tf.random_normal([n_input], seed=225)),
                }

            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['encoder_h1'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['encoder_h2'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['decoder_h1'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['decoder_h2'])

            # Building the encoder
            def encoder(x):
                # Encoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                               biases['encoder_b1']))

                layer_1 = layer_1 + layer_1

                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                               biases['encoder_b2']))
                return layer_2

            # Building the decoder
            def decoder(x):
                # Encoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                               biases['decoder_b1']))
                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                               biases['decoder_b2']))
                return layer_2

            # Construct model
            encoder_op = encoder(X) 			# 128 Features
            decoder_op = decoder(encoder_op)	# 784 Features

            # Prediction
            y_pred = decoder_op	# After
            # Targets (Labels) are the input data.
            y_true = X			# Before

            def loss_cal(y_true,y_pred):
                y_true2 = tf.gather_nd(y_true,id_0)
                y_pred2 = tf.gather_nd(y_pred,id_0)
                return tf.pow(y_true2 - y_pred2, 2)

            # Define loss and optimizer, minimize the squared error
            cost = tf.reduce_mean(loss_cal(y_true,y_pred))

            regularizer = tf.contrib.layers.l2_regularizer(0.01)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            cost += reg_term

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            # Launch the graph
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(times):
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={X: input_x})
                    if i%1000==0:
                        print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(c))
                print("Optimization Finished!")
                # # Applying encode and decode over test set
                A_result = sess.run(
                    y_pred, feed_dict={X: input_pre})
                for dd in range(len(select_diease_id)):
                    diease_id=select_diease_id[dd]
                    RMSE, MAE=test_loss(A_result[dd,:], ward_nor_list, yy, diease_id, input_dimension)

