import _pickle as cPickle, gzip, numpy
import numpy as np
import matplotlib.pyplot as plt
import time

# load data
f = gzip.open("/Users/joechen/Downloads/mnist.pkl.gz", 'rb')
train_set, valid_set, test_set = cPickle.load(f,encoding='latin1')
f.close()

train_4 = train_set[0][np.where(train_set[1] == 4)]
train_label4 = np.zeros((len(train_4), 1))
train_9 = train_set[0][np.where(train_set[1] == 9)]
train_label9 = np.ones((len(train_9), 1))

trainData = np.concatenate((train_4, train_9))
trainDataLabel = np.concatenate((train_label4, train_label9)) 
randomOrder = np.random.permutation(len(trainData))
trainData = trainData[randomOrder].T
trainDataLabel = trainDataLabel[randomOrder].T

test_4 = test_set[0][np.where(test_set[1] == 4)]
test_label4 = np.zeros((len(test_4), 1))
test_9 = test_set[0][np.where(test_set[1] == 9)]
test_label9 = np.ones((len(test_9), 1))

testData = np.concatenate((test_4, test_9))
testDataLabel = np.concatenate((test_label4, test_label9))
testData = testData.T
testDataLabel = testDataLabel.T


def first_derivative(trainDatas, lables, theta, lamda):
    numOfData = len(trainDatas[0])
    numOfFeature = len(theta)
    res = np.zeros((numOfFeature,1))
    for i in range(0, numOfData):
        Xi = np.insert(trainDatas[:,i], 0, 1).reshape(numOfFeature, 1)
        Yi = lables[0][i]
        tmp = Xi * 1 / (1 + np.exp(-np.transpose(theta) @ Xi)) - Yi * Xi
        res += tmp
    res += 2 * lamda * theta
    return res

def obj_fun(trainDatas, lables, theta, lamda):
    numOfData = len(trainDatas[0])
    numOfFeature = len(theta)
    res = 0
    for i in range(0, numOfData):
        Xi = np.insert(trainDatas[:,i], 0, 1).reshape(numOfFeature, 1)
        Yi = lables[0][i]
        tmp = 1 / (1 + np.exp(-np.transpose(theta) @ Xi))
        res += (Yi * np.log(tmp) + (1 - Yi) * np.log(1 - tmp))
    return (-res + lamda * theta.T @ theta)


##################train our theta using training data
numOfFeature = len(trainData) + 1
theta = np.zeros((numOfFeature, 1))
lamda = 5
lr = 1 * 10 ** (-4)
numOfIteration = 40
theta_list = []
loss_list = []
time_list = []
for i in range(0,numOfIteration):
#     first_dir = first_derivative(trainData, trainDataLabel, theta, lamda)
#     second_dir = second_derivative(trainData, theta, lamda)
#     inversed_second_dir = np.linalg.inv(second_dir);
    start = time.time()
    theta_derivative = first_derivative(trainData, trainDataLabel, theta, lamda)
    newTheta = theta - lr * theta_derivative
#     if i % 1000 == 0:
#         lr /= 2
#     count_right = testCase(test_data, test_label, thetaNew)
#     print(count_right / 1000)
    
#     theta = thetaNew
    theta = newTheta
    theta_list.append(theta)
    end = time.time()
    time_list.append(end - start)
#     if i % 250 == 0:
    print(i)
    print(np.linalg.norm(theta))
    print(time_list[-1])
    loss = obj_fun(trainData, trainDataLabel, theta, lamda)
    loss_list.append(loss)

# load z* we calculate previously
f = open("/Users/joechen/Desktop/545project/theta_start.txt", "r")
my_list = []

for line in f.readlines():
    my_list.append(float(line))

theta_start = np.array(my_list).reshape(len(theta_start), 1)

# calculate |z - z*|
res = []
for num in theta_list:
    res.append(np.linalg.norm(num - theta_start))
plt.plot(range(0, len(res)), res)

# calculate ratio
res_ratio = []
for i in range(1, len(res)):
    res_ratio.append(res[i] / res[i - 1])
plt.title("GD")
plt.ylabel("rate")
plt.xlabel("Iterations")    
plt.plot(range(0, 100), res_ratio[:100])