import numpy as np
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

data = np.loadtxt('crimerate.csv', delimiter=',')

sample = data[:,0:-1]
label = data[:,-1]

[n, p] = sample.shape

n_train = int(n*0.75)
n_test = int(n*0.25)

sample_train = sample[0:n_train,:]
sample_test = sample[n-n_test:,:]

label_train = label[0:n_train]
label_test = label[n-n_test:]

bet = np.random.rand(p)
no_coor_update = [1, 2, 3, 4, 5, 6, 7]
no_coor = 0
lamb = 10

non_ze_li = []
testing_errors = []

for num in no_coor_update:
    while no_coor < num:
        index = np.random.choice(bet.shape[0], 1, replace=False)
        if index[0] == 0:
            bet[0] = (-1/n_train) * np.sum(np.matmul(sample_train, bet) - label_train)

        if index[0] > 0:
            #non_ze = np.transpose(np.nonzero(bet))
            #[n_1, p_1] = non_ze.shape
            A_j = np.matmul(sample_train, bet) - label_train

            for i in range(p):
                if 2 * np.matmul(np.transpose(sample_train[:, i]), A_j).item() < -lamb:
                    a_1 = (-lamb) - (2 * np.matmul(np.transpose(sample_train[:, i]), A_j).item())
                    a_2 = 2 * np.matmul(np.transpose(sample_train[:, i]), sample_train[:, i]).item()
                    bet[i] = a_1/a_2
                    if bet[i] == math.inf:
                        bet[i] = 1
                elif 2 * np.matmul(np.transpose(sample_train[:, i]), A_j).item() > lamb:
                    b_1 = lamb - (2 * np.matmul(np.transpose(sample_train[:, i]), A_j).item())
                    b_2 = 2 * np.matmul(np.transpose(sample_train[:, i]), sample_train[:, i]).item()
                    bet[i] = b_1 / b_2
                    if bet[i] == math.inf:
                        bet[i] = 1
                elif np.absolute(2 * np.matmul(np.transpose(sample_train[:, i]), A_j)).item() <= lamb:
                    bet[i] = 0
                    if bet[i] == math.inf:
                        bet[i] = 1
        no_coor += 1
    count = 0
    for j in range(p):
        if bet[j] != 0.0:
            count += 1
    non_ze_li.append(count)

    pred_test = np.matmul(sample_test, bet)
    mse_test = mean_squared_error(label_test, pred_test)
    testing_errors.append(mse_test.item())

    no_coor = 0

plt.plot(no_coor_update, testing_errors, color='b', label='testing errors')
plt.xlabel("no_CD")
plt.ylabel("Errors")
plt.title("testing errors versus no_CD")
plt.legend()
plt.show()

plt.plot(no_coor_update, non_ze_li, color='r', label='non-zero')
plt.xlabel("no_CD")
plt.ylabel("non_zero")
plt.title("Non zero")
plt.legend()

plt.show()





