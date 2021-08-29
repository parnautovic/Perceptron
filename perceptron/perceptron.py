import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_path = 'hw1input.csv'
output_path = 'h21output.csv'
test_path = 'hw1test.csv'

input_data = pd.read_csv(input_path, sep = ' ', dtype='float')
print(input_data)

all_x = np.array(input_data.loc[:,'f(t)':])
all_d = np.array(input_data.loc[4:, 'f(t)']).transpose()

all_d2 = np.array(input_data.loc[9:, 'f(t)']).transpose()
all_d3 = np.array(input_data.loc[14:, 'f(t)']).transpose()


all_x1 = np.zeros((len(all_x)-4, 5), dtype='float')
all_x2 = np.zeros((len(all_x)-9, 10), dtype='float')
all_x3 = np.zeros((len(all_x)-14, 15), dtype='float')


for i in range(len(all_x1)):
    z = i
    for j in range(0, 5):
        #print(z)
        all_x1[i][j] = all_x[z]
        z+=1

for i in range(len(all_x2)):
    z = i
    for j in range(0, 10):
        #print(z)
        all_x2[i][j] = all_x[z]
        z+=1


for i in range(len(all_x3)):
    z = i
    for j in range(0, 15):
        #print(z)
        all_x3[i][j] = all_x[z]
        z+=1


for i in range(len(all_x1)):
  print(all_x1[i])
    #print(i)
all_inputs = -np.ones((len(all_x) - 4, 6), dtype='float')
all_inputs[:,1:] = all_x1

all_inputs2 = -np.ones((len(all_x) - 9, 11), dtype='float')
all_inputs2[:,1:] = all_x2

all_inputs3 = -np.ones((len(all_x) - 14, 16), dtype='float')
all_inputs3[:,1:] = all_x3

#for i in range(len(all_inputs2)):
  #  print(all_inputs2[i])
    #print(i)

def logsig(x):
    return 1/(1 + np.exp(-x))

def logsigder(x):
    s = logsig(x)
    return s * (1 - s)

def calculate_output(w_1, w_2, input_l):
  y_1 = input_l * w_1
  y_1 = logsig(np.sum(y_1, 1))
  y_1 = np.hstack(([-1], y_1))
  y_2 = logsig(np.sum(y_1 * w_2))
  return y_2

def mse(w_1, w_2, inputs, outputs):
  p = len(inputs)
  e = 0
  for i in range(p):
    err = calculate_output(w_1, w_2, inputs[i]) - outputs[i]
    e += 0.5 * err * err
  return e / p


confs = []
confs2 = []
confs3 = []
with open(output_path, 'w') as fout:
    fout.write('Training MSE_1 Epochs_1 MSE_2 Epochs_2 MSE_3 Epocs_3\n')
    eta = 0.1
    epsilon = 0.5 * 1e-6
    alfa = 0.8
    mse_per_epoch = []
    mse_per_epoch2 = []
    mse_per_epoch3 = []
    for i in range(1, 4):
        mse_per_epoch.append([])
        mse_per_epoch2.append([])
        mse_per_epoch3.append([])

        #TDNN_1
        w_1 = np.random.uniform(size=(10, 6))
        w_2 = np.random.uniform(size=(1,11))
        epoch = 0
        diff = 1
        error = mse(w_1, w_2, all_inputs, all_d)
        mse_per_epoch[i - 1].append(error)

        #TDNN_2
        w_1_2 = np.random.uniform(size=(15, 11))
        w_2_2 = np.random.uniform(size=(1,16))
        epoch2 = 0
        diff2 = 1
        error2 = mse(w_1_2, w_2_2, all_inputs2, all_d2)
        mse_per_epoch2[i-1].append(error2)


        #TDNN_3
        w_1_3 = np.random.uniform(size=(25, 16))
        w_2_3 = np.random.uniform(size=(1, 26))
        epoch3 = 0
        diff3 = 1
        error3 = mse(w_1_3, w_2_3, all_inputs3, all_d3)
        mse_per_epoch3[i-1].append(error3)

        while diff > epsilon:
            prev_error = error
            #w_prev2 = w_2
            #print(w_prev2)
            #w_prev1 = w_1
            for k in range(len(all_inputs)):
                #unapred
                i_1 = np.sum(all_inputs[k] * w_1, 1)
                y_1 = logsig(i_1)
                y_11 = np.hstack(([-1], y_1))
                i_2 = np.sum(y_11 * w_2, 1)
                y_2 = logsig(i_2)


                #unazad
                w_prev2 = w_2
                w_prev1 = w_1

                delta_2 = (all_d[k] - y_2) * logsigder(i_2)
                w_2 = w_2  + alfa * (w_2 - w_prev2) + eta * delta_2 * y_2
                delta_1 = - np.sum(delta_2 * w_2) * logsigder(i_1)
                w_1 = w_1 + alfa * (w_1 - w_prev1) + (eta * delta_1 * y_1)[np.newaxis].transpose()
            error = mse(w_1, w_2, all_inputs, all_d)
            epoch+=1
            diff =abs(error - prev_error)
            mse_per_epoch[i-1].append(error)
            if epoch % 10 == 0:
               print('TDNN1 Trainging %d epoch %d mse %.6f diff %f' % (i, epoch, error, diff))

        while diff2 > epsilon:

            prev_error2 = error2
            #w_prev2_2 = w_2_2
            #w_prev1_2 = w_1_2

            for k in range(len(all_inputs2)):
                #unapred
                i_1_2 = np.sum(all_inputs2[k] * w_1_2, 1)
                y_1_2 = logsig(i_1_2)
                y_11_2 = np.hstack(([-1], y_1_2))
                i_2_2 = np.sum(y_11_2 * w_2_2, 1)
                y_2_2 = logsig(i_2_2)

                w_prev2_2 = w_2_2
                w_prev1_2 = w_1_2

                #unazad
                delta_2_2 = (all_d2[k] - y_2_2) * logsigder(i_2_2)
                w_2_2 = w_2_2 + alfa * (w_2_2 - w_prev2_2) + eta * delta_2_2 * y_2_2
                delta_1_2 = - np.sum(delta_2_2 * w_2_2) * logsigder(i_1_2)
                w_1_2 = w_1_2 + alfa * (w_1_2 - w_prev1_2) + (eta * delta_1_2 * y_1_2)[np.newaxis].transpose()
            error2 = mse(w_1_2, w_2_2, all_inputs2, all_d2)
            epoch2+=1
            diff2 = abs(error2 - prev_error2)
            mse_per_epoch2[i - 1].append(error2)
            if epoch2 % 10 == 0:
               print('TDNN2 Trainging %d epoch %d mse %.6f diff %f' % (i, epoch2, error2, diff2))

        while diff3 > epsilon:
            prev_error3 = error3
            #w_prev2_3 = w_2_3
            #w_prev1_3 = w_1_3
            for k in range(len(all_inputs3)):
                # unapred
                i_1_3 = np.sum(all_inputs3[k] * w_1_3, 1)
                y_1_3 = logsig(i_1_3)
                y_11_3 = np.hstack(([-1], y_1_3))
                i_2_3 = np.sum(y_11_3 * w_2_3, 1)
                y_2_3 = logsig(i_2_3)

                w_prev2_3 = w_2_3
                w_prev1_3 = w_1_3

                # unazad
                delta_2_3 = (all_d3[k] - y_2_3) * logsigder(i_2_3)
                w_2_3 = w_2_3 + alfa * (w_2_3 - w_prev2_3) + eta * delta_2_3 * y_2_3
                delta_1_3 = - np.sum(delta_2_3 * w_2_3) * logsigder(i_1_3)
                w_1_3 = w_1_3 + alfa * (w_1_3 - w_prev1_3) + (eta * delta_1_3 * y_1_3)[np.newaxis].transpose()
            error3 = mse(w_1_3, w_2_3, all_inputs3, all_d3)
            epoch3 += 1
            diff3 = abs(error3 - prev_error3)
            mse_per_epoch3[i - 1].append(error3)
            if epoch3 % 10 == 0:
                print('TDNN3 Trainging %d epoch %d mse %.6f diff %f' % (i, epoch3, error3, diff3))
        confs.append((w_1,w_2))
        confs2.append((w_1_2, w_2_2))
        confs3.append((w_1_3, w_2_3))
        fout.write('%d %f %d %f %d %f %d\n' % (i, error, epoch, error2, epoch2, error3, epoch3))
print('Done.')

output_data = pd.read_csv(output_path, sep=' ')
test_data = pd.read_csv(test_path, sep=' ')

#TDNN1 Test
tdo = np.array(test_data.loc[:, 'f(t)'])
td = -np.ones((1,6), dtype=float)
p05 = all_d[91:96]
td[0,1:] = p05


output_val = []
relative_err = []
#print(confs)
for t in range(len(confs)):
    relative_err.append([])
    output_val.append([])
x = [td, td, td]

for k in range(len(tdo)):
    for t in range(len(confs)):
        w_1, w_2 = confs[t]
        output = calculate_output(w_1, w_2, x[t])
        update = -np.ones((1,6), dtype=float)
        update[:,1:5] = x[t][:,2:6]
        update[:,5] = output
        x[t] = update
        relative_err[t].append(abs((output-tdo[k])/tdo[k]))
        output_val[t].append(output)
        print('TDNN1 Primer %d iz treninga %d vrednost %f zeljena %f vrednost' % (k+1,t+1,output,tdo[k]))
mean_realtive_err = [sum(relative_err[k])/len(relative_err[k]) for k in range(len(relative_err))]
print('TDNN1 Relativne greske treninga', mean_realtive_err)

mean = [sum(output_val[k])/len(output_val[k]) for k in range(len(output_val))]
stdevsq = [[(output_val[k][j] - mean[k])**2 for j in range(len(output_val[k]))] for k in range(len(output_val))]
varince =[sum(stdevsq[k])/len(stdevsq[k]) for k in range(len(stdevsq))]
print('TDNN1 Varijansa treinga', varince)

meant = sum(tdo)/len(tdo)
stdevsqt = [(d - meant)**2 for d in tdo]
varincet = sum(stdevsqt)/len(stdevsqt)
print('TDNN1 Varijansa testa ', varincet)

#TDNN2 Test
td2 = -np.ones((1, 11), dtype=float)
p10 = all_d2[81:91]
td2[0,1:] = p10

output_val2 = []
relative_err2 = []
for t in range(len(confs2)):
    relative_err2.append([])
    output_val2.append([])

x2 = [td2,td2,td2]

for k in range(len(tdo)):
    for t in range(len(confs2)):
        w_1_2, w_2_2 = confs2[t]
        output2 = calculate_output(w_1_2, w_2_2, x2[t])
        update2 = -np.ones((1,11), dtype=float)
        update2[:,1:10] = x2[t][:,2:11]
        update2[:,10] = output2
        x2[t] = update2
        relative_err2[t].append(abs((output2 - tdo[k])/tdo[k]))
        output_val2[t].append(output2)
        print('TDNN2 Primer %d iz treninga %d vrednost %f zeljena vrednost %f' % (k+1, t+1, output2, tdo[k]))
mean_realtive_err2=[sum(relative_err2[k])/len(relative_err2[k]) for k in range(len(relative_err2))]
print('Relativne greske treninga ', mean_realtive_err2)

mean2 = [sum(output_val2[k])/len(output_val2[k]) for k in range(len(output_val2))]
stdevsq2 = [[(output_val2[k][j] - mean2[k])**2 for j in range(len(output_val2[k]))] for k in range(len(output_val2))]
varince2 =[sum(stdevsq2[k])/len(stdevsq2[k]) for k in range(len(stdevsq2))]
print('TDNN2 Varijansa treinga', varince2)

meant2 = sum(tdo)/len(tdo)
stdevsqt2 = [(d - meant2)**2 for d in tdo]
varincet2 = sum(stdevsqt2)/len(stdevsqt2)
print('TDNN2 Varijansa testa ', varincet2)

#TDNN3 Test
td3 = -np.ones((1, 16), dtype=float)
p15= all_d3[71:86]
td3[0,1:] = p15

output_val3 = []
relative_err3 = []
for t in range(len(confs3)):
    relative_err3.append([])
    output_val3.append([])

x3 = [td3,td3,td3]

for k in range(len(tdo)):
    for t in range(len(confs3)):
        w_1_3, w_2_3 = confs3[t]
        output3 = calculate_output(w_1_3, w_2_3, x3[t])
        update3 = -np.ones((1,16), dtype=float)
        update3[:,1:15] = x3[t][:,2:16]
        update3[:,15] = output3
        x3[t] = update3
        relative_err3[t].append(abs((output3 - tdo[k])/tdo[k]))
        output_val3[t].append(output3)
        print('TDNN3 Primer %d iz treninga %d vrednost %f zeljena vrednost %f' % (k+1, t+1, output3, tdo[k]))
mean_realtive_err3=[sum(relative_err3[k])/len(relative_err3[k]) for k in range(len(relative_err3))]
print('Relativne greske treninga ', mean_realtive_err3)

mean3 = [sum(output_val3[k])/len(output_val3[k]) for k in range(len(output_val3))]
stdevsq3 = [[(output_val3[k][j] - mean3[k])**2 for j in range(len(output_val3[k]))] for k in range(len(output_val3))]
varince3 =[sum(stdevsq3[k])/len(stdevsq3[k]) for k in range(len(stdevsq3))]
print('TDNN3 Varijansa treinga', varince3)

meant3 = sum(tdo)/len(tdo)
stdevsqt3 = [(d - meant3)**2 for d in tdo]
varincet3 = sum(stdevsqt3)/len(stdevsqt3)
print('TDNN3 Varijansa testa ', varincet3)

mse_out_1 = np.array(output_data.loc[:,'MSE_1'])

print(mse_out_1)
min = 1

mse_min_1 = mse_out_1.min()


ts = sorted(tuple(enumerate(mse_per_epoch)), key=lambda x: -len(x[1]))
l1 = ts[0][1]
l2 = ts[1][1]
l3 = ts[2][1]
epocs = tuple(range(len(l1)))
plt.plot(tuple(range(len(l1))), l1, 'r.')
plt.plot(tuple(range(len(l2))), l2, 'b.')
plt.plot(tuple(range(len(l3))), l3, 'm.')

plt.show()


