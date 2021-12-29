import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = np.loadtxt('crimerate.csv', delimiter=',')

sample = data[:, 0:-1]
label = data[:, -1]

[n, p] = sample.shape

n_train = int(n*0.75)
#print(n_train)
n_test = int(n*0.25)

sample_train = sample[0:n_train, :]
sample_test = sample[n-n_test:, :]

label_train = label[0:n_train]
label_test = label[n-n_test:]

m = [n_train, int(n_train*0.90), int(n_train*0.80), int(n_train*0.70), int(n_train*0.60), int(n_train*0.50), int(n_train*0.40), int(n_train*0.30), int(n_train*0.20), int(n_train*0.10)]
lamb = 10
training_errors = []
testing_errors = []

for m_value in m:
    new_sample_train = sample_train[0:m_value,:]
    new_label_train = label_train[0:m_value]
    new = np.matmul(np.square(new_sample_train), np.transpose(np.square(new_sample_train)))
    gram = np.matmul(np.transpose(new), new)
    #print(np.shape(gram))

    new_mat = np.matmul(np.transpose(gram), gram) + (lamb * gram)
    #print(np.shape(new_mat))
    inver = np.linalg.inv(new_mat)
    #print(np.shape(inver))
    alpha = np.matmul(np.matmul(inver, np.transpose(gram)), new_label_train)
    phi_xi = np.square(new_sample_train)
    bet = np.matmul(np.transpose(phi_xi), alpha)
    #print(np.shape(bet))
    #print(np.shape(new_sample_train))
    #print(np.shape(new_label_train))

    pred_train = np.matmul(new_sample_train, bet)
    mse_train = mean_squared_error(new_label_train, pred_train)
    training_errors.append(mse_train.item())

    pred_test = np.matmul(sample_test, bet)
    mse_test = mean_squared_error(label_test, pred_test)
    testing_errors.append(mse_test.item())

plt.plot(m, training_errors, color='b', label='training errors')
plt.plot(m, testing_errors, color='r', label='testing errors')

plt.xlabel("m_value")
plt.ylabel("Errors")
plt.title("Training/testing errors versus m_value")

plt.legend()
plt.show()