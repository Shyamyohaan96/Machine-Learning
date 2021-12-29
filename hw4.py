import numpy as np
import matplotlib.pyplot as plt

def sigmoidfunction(x):
    return 1.0/(1.0+np.exp(-x))


data = np.loadtxt('crimerate.csv', delimiter=',')

sample = data[:, 0:-1]
label = data[:, -1]

for i in range(label.shape[0]):
    if label[i] > 0.15:
        label[i] = 1
    else:
        label[i] = 0

[n, p] = sample.shape

n_train = int(n*0.50)
n_test = int(n*0.50)

sample_train = sample[0:n_train, :]
sample_test = sample[n-n_test:, :]

label_train = label[0:n_train]
label_test = label[n-n_test:]

w = np.zeros(p)
b = 0
no_of_updates = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
l_rate = 0.001
test_error = []
test_error_2 = []

#Gradient descent
for iter in no_of_updates:
    for j in range(iter):
        li = np.dot(sample_train, w) + b
        pre_y = sigmoidfunction(li)
        der_w = (1.0/sample_train.shape[0]) * np.dot(np.transpose(sample_train), (pre_y-label_train))
        der_b = (1.0/sample_train.shape[0]) * np.sum(pre_y-label_train)
        w -= l_rate * der_w
        b -= l_rate * der_b

    te_li = np.dot(sample_test, w) + b
    te_pre_y = sigmoidfunction(te_li)
    for i in range(te_pre_y.shape[0]):
        if te_pre_y[i] > 0.5:
            te_pre_y[i] = 1
        else:
            te_pre_y[i] = 0

    error = np.sum(label_test == te_pre_y)/ label_test.shape[0]
    test_error.append(1.0 - error)

#Newton method
w = np.zeros(p)
lrate = 10
for iter in no_of_updates:
    for j in range(iter):
        li = np.dot(sample_train, w) + b
        pre_y = sigmoidfunction(li)
        der_w = (1.0/sample_train.shape[0]) * np.dot(np.transpose(sample_train), (pre_y-label_train))
        her_mat = (1.0/sample_train.shape[0]) * np.dot(np.transpose(sigmoidfunction(li)), (1.0 - sigmoidfunction(li))) * np.dot(np.transpose(sample_train), sample_train)
        der_w_new =  np.dot(np.linalg.inv(her_mat), der_w)
        der_b = (1.0 / sample_train.shape[0]) * np.sum(pre_y - label_train)
        w -= lrate* der_w_new
        b -= lrate* der_b

    te_li = np.dot(sample_test, w) + b
    te_pre_y = sigmoidfunction(te_li)
    for i in range(te_pre_y.shape[0]):
        if te_pre_y[i] > 0.5:
            te_pre_y[i] = 1
        else:
            te_pre_y[i] = 0

    error = np.sum(label_test == te_pre_y)/ label_test.shape[0]
    test_error_2.append(1.0 - error)

plt.plot(no_of_updates, test_error, color='r', label='Gradient descent')
plt.plot(no_of_updates, test_error_2, color='b', label='Newton Method')

plt.xlabel("No of updates")
plt.ylabel("Errors")
plt.title("Testing errors")

plt.legend()
plt.show()
