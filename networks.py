# Imported Functions
from lattice_importer import importLattices as importer # Importer for data

from scipy.optimize import curve_fit # Function used to fit custom curves to data
from os.path import isfile
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import random as rn
import numpy as np

import os

# Custom Functions
def line(x, m, b):
    return x * m + b

def quadratic(x, a, b, c):
    return a * x**2.0 + b * x + c

def tanh_fit(x, k, b):
    return 0.5 * np.tanh(k*x + b) + 0.5

def sigma_fit(x, k, b):
    return np.exp(k*x) / (b + np.exp(k*x))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

def neuralNet(L, numb_batches, test_set, training_set=None, validation_set=None, existing_model=None,
            numb_conv1=None, numb_conv2=None, numb_relu1=128, numb_relu2=None, batch_size=100, step_size=0.1,
            field_size=2, reg_param=None, report_freq=100, pooling=False):
    x = tf.placeholder(tf.float32, shape=[None, 3*L*L*L])
    if numb_conv1 != None:
        x_image = tf.reshape(x, [-1, L, L, L, 3])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    #limit = tf.constant(0.05,shape=[1])
    
    step_size = tf.Variable(step_size)
    reg_param = tf.constant(reg_param)
    reduce_step_size = step_size.assign(step_size - 0.000025)

    if numb_conv1 != None:
        W_conv1 = weight_variable([field_size, field_size, field_size, 3, numb_conv1])
        b_conv1 = bias_variable([numb_conv1])
        if numb_conv2 != None:
            W_conv2 = weight_variable([field_size, field_size, field_size, numb_conv1, numb_conv2])
            b_conv2 = bias_variable([numb_conv2])
    if numb_relu1 != None:
        if numb_conv1 != None:
            if numb_conv2 != None:
                if pooling:
                    W_relu1 = weight_variable([L*L*L/64*numb_conv2, numb_relu1])
                else:
                    W_relu1 = weight_variable([L*L*L*numb_conv2, numb_relu1])
                b_relu1 = bias_variable([numb_relu1])
            else:
                if pooling:
                    W_relu1 = weight_variable([L*L*L/8*numb_conv1, numb_relu1])
                else:
                    W_relu1 = weight_variable([L*L*L*numb_conv1, numb_relu1])
                b_relu1 = bias_variable([numb_relu1])
        else:
            W_relu1 = weight_variable([3*L*L*L, numb_relu1])
            b_relu1 = bias_variable([numb_relu1])
        if numb_relu2 != None:
            W_relu2 = weight_variable([numb_relu1, numb_relu2])
            b_relu2 = bias_variable([numb_relu2])
            W_soft = tf.Variable(tf.zeros([numb_relu2,2]))
            b_soft = tf.Variable(tf.zeros([2]))
        else:
            W_soft = tf.Variable(tf.zeros([numb_relu1,2]))
            b_soft = tf.Variable(tf.zeros([2]))
    else:
        if pooling:
            W_soft = tf.Variable(tf.zeros([L*L*L/8*numb_conv1,2]))
        else:
            W_soft = tf.Variable(tf.zeros([L*L*L*numb_conv1,2]))
        b_soft = tf.Variable(tf.zeros([2]))

    if pooling:
        if numb_conv1 != None:
            a_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
            a_pool1 = max_pool_2x2x2(a_conv1)
            if numb_conv2 != None:
                a_conv2 = tf.nn.relu(conv3d(a_pool1, W_conv2) + b_conv2)
                a_pool2 = max_pool_2x2x2(a_conv2)
                a_pool2_flat = tf.reshape(a_pool2, [-1, L*L*L/64*numb_conv2])
            else:
                a_pool1_flat = tf.reshape(a_pool1, [-1, L*L*L/8*numb_conv1])
    else:
        if numb_conv1 != None:
            a_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
            if numb_conv2 != None:
                a_conv2 = tf.nn.relu(conv3d(a_conv1, W_conv2) + b_conv2)
                a_conv2_flat = tf.reshape(a_conv2, [-1, L*L*L*numb_conv2])
            else:
                a_conv1_flat = tf.reshape(a_conv1, [-1, L*L*L*numb_conv1])
    if numb_relu1 != None:
        if numb_conv1 != None:
            if numb_conv2 != None:
                a_relu1 = tf.nn.relu(tf.matmul(a_pool2_flat, W_relu1) + b_relu1)
            else:
                a_relu1 = tf.nn.relu(tf.matmul(a_pool1_flat, W_relu1) + b_relu1)
        else:
            a_relu1 = tf.nn.relu(tf.matmul(x, W_relu1) + b_relu1)
        # keep_prob = tf.placeholder(tf.float32)
        # a_relu1_drop = tf.nn.dropout(a_relu1, keep_prob)
        if numb_relu2 != None:
            a_relu2 = tf.nn.relu(tf.matmul(a_relu1, W_relu2) + b_relu2)
            a_soft = tf.nn.softmax(tf.matmul(a_relu2,W_soft) + b_soft)
        else:
            a_soft = tf.nn.softmax(tf.matmul(a_relu1,W_soft) + b_soft)
    elif pooling:
        if numb_conv2 != None:
            a_soft = tf.nn.softmax(tf.matmul(a_pool2_flat,W_soft) + b_soft)
        else:
            a_soft = tf.nn.softmax(tf.matmul(a_pool1_flat,W_soft) + b_soft)
    else:
        if numb_conv2 != None:
            a_soft = tf.nn.softmax(tf.matmul(a_conv2_flat,W_soft) + b_soft)
        else:
            a_soft = tf.nn.softmax(tf.matmul(a_conv1_flat,W_soft) + b_soft)

    # calc_reg = reg_param / 2.0 * tf.reduce_mean(tf.square(W_relu1))
    # avg_weight = tf.reduce_mean(tf.square(W_relu1))

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a_soft), reduction_indices=[1]))
    # step_size = 1.0/float(L*L)
    train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(a_soft,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    formatted_test_set = [[d['values'] for d in test_set],[d['phase'] for d in test_set]]
    init_acc = sess.run(accuracy, feed_dict={x: formatted_test_set[0], y: formatted_test_set[1]})
    print "Initial Te.Accuracy: " + str(init_acc)


    if existing_model == None:
        formatted_validation_set = [[d['values'] for d in validation_set],[d['phase'] for d in validation_set]]
        model_acc = sess.run(accuracy, feed_dict={x: formatted_validation_set[0], y: formatted_validation_set[1]})
        v_accs = []
        train_accs = []
        v_accs.append(model_acc)
        print "Validation Accuracy " + str(model_acc)

        index = 0
        counter = 0
        #print "Step Size: " + str(sess.run(step_size))
        for i in range(numb_batches):
            counter += 1
            batch = [[],[]]
            if (index + batch_size) > len(training_set):
                rn.shuffle(training_set)
                index = 0
            for j in range(index,index+batch_size):
                batch[0].append(training_set[j]['values'])
                batch[1].append(training_set[j]['phase'])

            if (counter % report_freq) == 0:
                update_acc = sess.run(accuracy, feed_dict={x: formatted_validation_set[0], y: formatted_validation_set[1]})
                train_acc = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
                v_accs.append(update_acc)
                train_accs.append(train_acc)
                # if abs(update_acc - model_acc) < 0.0005:
                #     print "Validation Accuracy " + str(update_acc)
                #     break
                # else:
                    #print "Batch " + str(counter) + " Step Size: " + str(sess.run(step_size)) + " Accuracy: " + str(update_acc)
                #print "Regularization Term: " + str(sess.run(calc_reg)) + " Average Weight: " + str(sess.run(avg_weight)) + " Cross Entropy Term: " + str(sess.run(cross_entropy, feed_dict={x: [batch[0][0]], y:[batch[1][0]]}))
                print "Batch: " + str(counter) + " Step Size: " + str(sess.run(step_size)) + " V.Accuracy: " + str(update_acc) + " Tr.Accuracy: " + str(train_acc)
                model_acc = update_acc
            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

            # sess.run(reduce_step_size)

            index = index + batch_size

        final_acc = sess.run(accuracy, feed_dict={x: formatted_test_set[0], y: formatted_test_set[1]})
        print "Final Te.Accuracy: " + str(final_acc)

        output = sess.run(a_soft, feed_dict={x: formatted_test_set[0]})

        temps = [d['T'] for d in test_set]
        low_points = [o[0] for o in output]
        high_points = [o[1] for o in output]

        sort_list = []
        for i in range(len(temps)):
            sort_dict = {}
            sort_dict['temp'] = temps[i]
            sort_dict['low'] = low_points[i]
            sort_dict['high'] = high_points[i]
            sort_list.append(sort_dict)

        sorted_list = sorted(sort_list, key=lambda k: k['temp'])

        temps = []
        low_points = []
        high_points = []
        for sorted_dict in sorted_list:
            temps.append(sorted_dict['temp'])
            low_points.append(sorted_dict['low'])
            high_points.append(sorted_dict['high'])

        if L > 9:
            save_path = saver.save(sess, "network_data/TC-D3-L" + str(L) + "/model.ckpt")
        else:
            save_path = saver.save(sess, "network_data/TC-D3-L0" + str(L) + "/model.ckpt")

        return init_acc, final_acc, train_accs, v_accs, temps, low_points, high_points

    else:
        print existing_model
        saver.restore(sess, existing_model)
        final_acc = sess.run(accuracy, feed_dict={x: formatted_test_set[0], y: formatted_test_set[1]})
        print "Final Te.Accuracy: " + str(final_acc)

        output = sess.run(a_soft, feed_dict={x: formatted_test_set[0]})

        temps = [d['T'] for d in test_set]
        low_points = [o[0] for o in output]
        high_points = [o[1] for o in output]

        sort_list = []
        for i in range(len(temps)):
            sort_dict = {}
            sort_dict['temp'] = temps[i]
            sort_dict['low'] = low_points[i]
            sort_dict['high'] = high_points[i]
            sort_list.append(sort_dict)

        sorted_list = sorted(sort_list, key=lambda k: k['temp'])

        temps = []
        low_points = []
        high_points = []
        for sorted_dict in sorted_list:
            temps.append(sorted_dict['temp'])
            low_points.append(sorted_dict['low'])
            high_points.append(sorted_dict['high'])

        return init_acc, final_acc, temps, low_points, high_points

def plot_results(L, temps, low_points, high_points, smooth=False, smooth_numb=3, smooth_semi_range=50):
    if smooth:
        for k in range(smooth_numb):
            for j in range(smooth_semi_range,(len(wil)-smooth_semi_range)):
                for m in range(-smooth_semi_range,smooth_semi_range+1):
                    if m != 0:
                        wil[j] += wil[j+m]
                        para[j] += para[j+m]
                wil[j] = wil[j] / float(2*smooth_semi_range + 1)
            wil = wil[smooth_semi_range:-smooth_semi_range]

        reducer = smooth_semi_range * smooth_numb
        current_temps = temps[reducer:-reducer]
    else:
        current_temps = temps

    # start_pars = [1.0, -2.269]
    # wpars, wpcov = curve_fit(tanh_fit, current_temps, wil, p0=start_pars, maxfev=10000)

    # print fpars

    # wil_curve = []
    # for T in current_temps:
    #     wil_curve.append(tanh_fit(T, fpars[0], fpars[1]))


    plt.figure(L)
    # plt.plot(current_temps, wil_curve, 'k-')
    plt.title('CNN Output for L=' + str(L) + ' 3D Toric Code')
    plt.xlabel('Temperature')
    plt.ylabel('Activation')
    plt.plot(current_temps, high_points, 'r.', label='High T Phase')
    plt.plot(current_temps, low_points, 'b.', label='Low T Phase')
    plt.legend(loc=1, ncol=1)
    plt.savefig('results/TC-D3-L' + str(L), bbox_inches='tight')

# Ls = [16]
# for L in Ls:
#     path = "network_data/TC_L" + str(L) + "/model.ckpt"
#     if isfile(path):
#         test_set = importer('TC-D2-L' + str(L) + '/', test_only=True)
#         temps,wil = neuralNet(L, 1000, test_set, existing_model=path)
#     else:
#         training_set,validation_set,test_set = importer('TC-D2-L' + str(L) + '/')
#         temps,wil = neuralNet(L, 1000, test_set, training_set=training_set, validation_set=validation_set)
#     plot_results(L,temps,wil,smooth=False)



# trial_number = 1
# created = False
# while not created:
#     try:
#         directory = 'acc_plots/' + str(trial_number) + '/'
#         os.makedirs(directory)
#         created = True
#     except(OSError):
#         trial_number += 1

# ### Double Loop ###

# # L = 8
# # # numb_layer1_list = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# # numb_layer1 = 2048
# # step_size = 0.005
# # training_limits = [1,10,100,1000,8550]
# # # numb_layer2_list = [64, 128, 256, 512, 1024, 2048]
# # #step_sizes = [0.005,0.0001]
# # number_of_batches = 10000
# # report_freq = 200
# # numb_layer2 = None
# # reg_param = 1
# # plot_colours = ['b','c','g','y','r','m']
# # # accuracy_points = []
# # # test_accs = []

# # if L > 9:
# #     training_set,validation_set,test_set = importer('TC-D3-L' + str(L) + '/')
# # else:
# #     training_set,validation_set,test_set = importer('TC-D3-L0' + str(L) + '/')
# # for numb_layer1 in numb_layer1_list:
# #     train_accs_list = []
# #     val_accs_list = []
# #     for step_size in step_sizes:
# #         train_accs = []
# #         val_accs = []
# #         batches = range(0,number_of_batches+report_freq,report_freq)
# #         trunc_batches = batches[1:]
# #         print "Neurons: " + str(numb_layer1) + " Step Size: " + str(step_size)
# #         init_acc,final_acc,train_accs,val_accs = neuralNet(L, number_of_batches, test_set, training_set=training_set, validation_set=validation_set, step_size=step_size, numb_relu1=numb_layer1, numb_relu2=numb_layer2, numb_conv1=None, numb_conv2=None, reg_param=reg_param, report_freq=report_freq)
# #         train_accs_list.append(train_accs)
# #         val_accs_list.append(val_accs)
    
# #     fig,ax = plt.subplots(2)
# #     ax[0].set_title(str(numb_layer1) + ' Neuron FCN V. Step Size Accuracy vs. Batch')
# #     for i in range(len(step_sizes)):
# #         ax[0].plot(trunc_batches,train_accs_list[i], plot_colours[i] + '-', label=str(step_sizes[i]))

# #     ax[0].legend(loc=8, ncol=6, mode="expand", borderaxespad=0.)

# #     ax[1].set_title('Validation Accuracy')
# #     for i in range(len(step_sizes)):
# #         ax[1].plot(batches,val_accs_list[i], plot_colours[i] + '-', label=str(step_sizes[i]))

    
# #     plt.savefig(directory + str(numb_layer1))

# ### Single Loop ###

# L = 8
# # numb_layer1_list = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# numb_layer1 = None
# numb_layer2 = None
# #numb_conv1 = 8
# numb_conv1_list = [1,2,4,8]
# numb_conv2 = None
# field_size = 2
# # training_limits = [1,10,100,1000,8550]
# # numb_layer2_list = [64, 128, 256, 512, 1024, 2048]
# # step_size = 0.1
# step_size = 0.075
# number_of_batches = 2000
# report_freq = 50
# reg_param = 1
# plot_colours = ['b','c','g','y','r','m']
# # accuracy_points = []
# # test_accs = []
# if L > 9:
#     path = "network_data/TC-D3-L" + str(L) + "/model-void.ckpt"
# else:
#     path = "network_data/TC-D3-L0" + str(L) + "/model-void.ckpt"

# if isfile(path):
#     if L > 9:
#         test_set = importer('TC-D3-L' + str(L) + '/', test_only=True)
#     else:
#         test_set = importer('TC-D3-L0' + str(L) + '/', test_only=True)
#     existing_model = path
#     for numb_conv1 in numb_conv1_list:
#         print "Conv 1: " + str(numb_conv1)
#         init_acc, final_acc, temps, low_points, high_points = neuralNet(L, number_of_batches, test_set, step_size=step_size, numb_relu1=numb_layer1, numb_relu2=numb_layer2, numb_conv1=numb_conv1, numb_conv2=numb_conv2, reg_param=reg_param, report_freq=report_freq, field_size=field_size, existing_model=path, pooling=False)
#         plot_results(L, temps, low_points, high_points, smooth=False)
# else:
#     if L > 9:
#         training_set, validation_set, test_set = importer('TC-D3-L' + str(L) + '/')
#     else:
#         training_set, validation_set, test_set = importer('TC-D3-L0' + str(L) + '/')
    
#     train_accs_list = []
#     val_accs_list = []
#     for numb_conv1 in numb_conv1_list:
#         train_accs = []
#         val_accs = []
#         batches = range(0,number_of_batches+report_freq,report_freq)
#         trunc_batches = batches[1:]
#         print "Conv 1: " + str(numb_conv1)
#         init_acc, final_acc, train_accs, val_accs, temps, low_points, high_points = neuralNet(L, number_of_batches, test_set, training_set=training_set, validation_set=validation_set, step_size=step_size, numb_relu1=numb_layer1, numb_relu2=numb_layer2, numb_conv1=numb_conv1, numb_conv2=numb_conv2, reg_param=reg_param, report_freq=report_freq, field_size=field_size, existing_model=None)
#         plot_results(L, temps, low_points, high_points, smooth=False)
#         train_accs_list.append(train_accs)
#         val_accs_list.append(val_accs)

#     fig,ax = plt.subplots(2)
#     ax[0].set_title(str(numb_conv1) + '-' + str(numb_conv2) + '-' + str(numb_layer1) + '-' + str(numb_layer2) + ' CNN V. Step Size Accuracy vs. Batch')
#     for i in range(len(numb_conv1_list)):
#         ax[0].plot(trunc_batches,train_accs_list[i], plot_colours[i] + '-', label=str(numb_conv1_list[i]))

#     ax[0].legend(loc=8, ncol=6, mode="expand", borderaxespad=0.)

#     ax[1].set_title('Validation Accuracy')
#     for i in range(len(numb_conv1_list)):
#         ax[1].plot(batches,val_accs_list[i], plot_colours[i] + '-', label=str(numb_conv1_list[i]))


#     plt.savefig(directory + str(numb_conv1))


# # fig = plt.figure(0)
# # plt.title('Final Accuracy for V. Sized 1st Layer at V. of 2nd Layers')
# # plt.xlabel('Number of Neurons in Second Layer')
# # plt.ylabel('Accuracy')
# # for i in range(len(test_accs_list)):
# #     plt.plot(numb_layer2_list,test_accs_list[i], plot_colours[i] + '-', label=str(numb_layer1_list[i]))
# # plt.plot([0,2048],[init_acc, init_acc], 'k-')
# # plt.legend(loc=8, ncol=6, mode="expand", borderaxespad=0.)
# # plt.savefig(directory + 'test-accs')


# ### Single Training ###
# # L = 8
# # number_of_batches = 1000
# # report_freq = 100
# # batches = range(0,number_of_batches+report_freq,report_freq)
# # trunc_batches = range(report_freq,number_of_batches+report_freq,report_freq)
# # if L > 9:
# #     training_set,validation_set,test_set = importer('TC-D3-L' + str(L) + '/')
# # else:
# #     training_set,validation_set,test_set = importer('TC-D3-L0' + str(L) + '/')
# # init_acc,final_acc,train_accs,v_accs = neuralNet(L, number_of_batches, test_set, training_set=training_set, validation_set=validation_set, step_size=0.01, numb_relu1=1024, numb_relu2=None, numb_conv1=None, numb_conv2=None, reg_param=1.0, batch_size=100, report_freq=report_freq)

# # print "Init Acc: " + str(init_acc) + " Final Acc: " + str(final_acc)
# # fig,ax = plt.subplots(2)
# # ax[0].plot(trunc_batches,train_accs, 'b-', label="Train")
# # ax[1].plot(batches,v_accs, 'r-', label="Val")

# # plt.show()







trial_number = 1
created = False
while not created:
    try:
        directory = 'acc_plots/' + str(trial_number) + '/'
        os.makedirs(directory)
        created = True
    except(OSError):
        trial_number += 1

L = 8
numb_layer1 = None
numb_layer2 = None
numb_conv1 = 4
numb_conv2 = None
field_size = 2
step_size = 0.075
number_of_batches = 8000
report_freq = 50
reg_param = 1
plot_colours = ['b','c','g','y','r','m']
if L > 9:
    path = "network_data/TC-D3-L" + str(L) + "/model-void.ckpt"
else:
    path = "network_data/TC-D3-L0" + str(L) + "/model-void.ckpt"

if isfile(path):
    if L > 9:
        test_set = importer('TC-D3-L' + str(L) + '/', test_only=True)
    else:
        test_set = importer('TC-D3-L0' + str(L) + '/', test_only=True)
    existing_model = path
    # for numb_conv1 in numb_conv1_list:
    #     print "Conv 1: " + str(numb_conv1)
    #     init_acc, final_acc, temps, low_points, high_points = neuralNet(L, number_of_batches, test_set, step_size=step_size, numb_relu1=numb_layer1, numb_relu2=numb_layer2, numb_conv1=numb_conv1, numb_conv2=numb_conv2, reg_param=reg_param, report_freq=report_freq, field_size=field_size, existing_model=path, pooling=False)
    #     plot_results(L, temps, low_points, high_points, smooth=False)
else:
    if L > 9:
        training_set, validation_set, test_set = importer('TC-D3-L' + str(L) + '/')
    else:
        training_set, validation_set, test_set = importer('TC-D3-L0' + str(L) + '/')
    
    train_accs = []
    val_accs = []
    batches = range(0,number_of_batches+report_freq,report_freq)
    trunc_batches = batches[1:]
    init_acc, final_acc, train_accs, val_accs, temps, low_points, high_points = neuralNet(L, number_of_batches, test_set, training_set=training_set, validation_set=validation_set, step_size=step_size, numb_relu1=numb_layer1, numb_relu2=numb_layer2, numb_conv1=numb_conv1, numb_conv2=numb_conv2, reg_param=reg_param, report_freq=report_freq, field_size=field_size, existing_model=None)
    plot_results(L, temps, low_points, high_points, smooth=False)

    fig,ax = plt.subplots(2)
    ax[0].set_title(str(numb_conv1) + '-' + str(numb_conv2) + '-' + str(numb_layer1) + '-' + str(numb_layer2) + ' CNN V. Step Size Accuracy vs. Batch')
    ax[0].plot(trunc_batches,train_accs, 'b-')

    ax[1].set_title('Validation Accuracy')
    ax[1].plot(batches,val_accs, 'b-')


    plt.savefig(directory + str(numb_conv1))