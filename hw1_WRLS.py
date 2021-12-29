import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = np.loadtxt('crimerate.csv', delimiter=',')

sample = data[:,0:-1]
label = data[:,-1]

[n,p] = sample.shape

n_train = int(n*0.20)
n_test = int(n*0.25)

sample_train = sample[0:n_train,:]
sample_test = sample[n-n_test:,:]

label_train = label[0:n_train]
label_test = label[n-n_test:]

lamb = [0.1, 1, 10, 50, 100]
training_errors = []
testing_errors = []
iden_with_zero = np.identity(p)
iden_with_zero[0,0] = 0
weights = np.identity(n_train)
for la in lamb:
    new_mat = np.matmul(np.matmul(np.transpose(sample_train), weights), sample_train) + la * iden_with_zero
    inver = np.linalg.inv(new_mat)
    bet = np.matmul(np.matmul(np.matmul(inver, np.transpose(sample_train)), weights), label_train)

    pred_train = np.matmul(sample_train,bet)
    mse_train = mean_squared_error(label_train, pred_train)
    training_errors.append(mse_train.item())

    pred_test = np.matmul(sample_test, bet)
    mse_test = mean_squared_error(label_test, pred_test)
    testing_errors.append(mse_test.item())

plt.plot(lamb, training_errors, color='b', label='training errors')
plt.plot(lamb, testing_errors, color='r', label='testing errors')

plt.xlabel("Lambda")
plt.ylabel("Errors")
plt.title("Training/testing errors versus Lambda")

plt.legend()
plt.show()

########################################### Part 2 ########################################################
### Based on the the previous graph it is inferred that taking lambda around 15 or 16 gave almost same training and testing error####
### So for this exercise lambda is considered to be 16 #####

n_train_2 = int(n*0.75)
n_test_2 = int(n*0.25)

sample_train_2 = sample[0:n_train_2,:]
sample_test_2 = sample[n-n_test_2:,:]

label_train_2 = label[0:n_train_2]
label_test_2 = label[n-n_test_2:]

mino_li_sample = []
mino_li_label = []
non_mino_li_sample = []
non_mino_li_label = []

for i in range(n_test_2):
    if sample_test_2[i, 0] < 0.02:
        mino_li_sample.append(list(sample_test_2[i, :]))
        mino_li_label.append(label_test_2[i])
    else:
        non_mino_li_sample.append(list(sample_test_2[i, :]))
        non_mino_li_label.append(label_test_2[i])

minority_test_sample = np.array(mino_li_sample)
minority_test_label = np.array(mino_li_label)

non_minority_test_sample = np.array(non_mino_li_sample)
non_minority_test_label = np.array(non_mino_li_label)

new_weights = [5, 10, 50]
weights_2 = np.identity(n_train_2)

lamb_2 = 16
final_li =[]

for we in new_weights:
    temp = []
    for j in range(n_train_2):
        if sample_train_2[j, 0] < 0.02:
            weights_2[j, j] = we

    new_mat_2 = np.matmul(np.matmul(np.transpose(sample_train_2), weights_2), sample_train_2) + lamb_2 * iden_with_zero
    inver_2 = np.linalg.inv(new_mat_2)
    bet_2 = np.matmul(np.matmul(np.matmul(inver_2, np.transpose(sample_train_2)), weights_2), label_train_2)

    pred_train_2 = np.matmul(sample_train_2, bet_2)
    mse_train_2 = mean_squared_error(label_train_2, pred_train_2)

    pred_all_test = np.matmul(sample_test_2, bet_2)
    mse_test_all = mean_squared_error(label_test_2, pred_all_test)
    temp.append(mse_test_all.item())

    pred_min_test = np.matmul(minority_test_sample, bet_2)
    mse_min_test = mean_squared_error(minority_test_label, pred_min_test)
    temp.append(mse_min_test.item())

    pred_non_min_test = np.matmul(non_minority_test_sample, bet_2)
    mse_non_min_test = mean_squared_error(non_minority_test_label, pred_non_min_test)
    temp.append(mse_non_min_test.item())

    final_li.append(temp)

tab = np.array(final_li)
tab = np.transpose(tab)

print(tab)






















