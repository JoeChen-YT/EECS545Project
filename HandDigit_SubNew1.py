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

##############define the method for 1st derivative
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

##############define the method for 2nd derivative
def second_derivative(trainDatas, theta, lamda):
    numOfData = len(trainDatas[0])
    print(numOfData)
    numOfFeature = len(theta)
    res = np.zeros((numOfFeature, numOfFeature))
    randomPer = np.random.permutation(numOfData)
    randomPerTrainDatas = trainDatas[:,randomPer]
    for i in range(0, 6000):
        Xi = np.insert(randomPerTrainDatas[:,i], 0, 1).reshape(numOfFeature, 1)
        expValue = np.exp(-np.transpose(theta) @ Xi)
        tmp = Xi @ np.transpose(Xi) * expValue / ((1 + expValue) ** 2)
        res += tmp
    res += np.identity(numOfFeature) * 2 * lamda
    return res

#############define the method for test method
def testCase(testDatas, testLable, theta):
    count_right = 0
    numOfDatas = len(testDatas[0])
    numOfFeature = len(theta)
    for i in range(0,numOfDatas):
        Xi = np.insert(testDatas[:,i],0,1).reshape(numOfFeature, 1)
        expValue = np.exp(-np.transpose(theta) @ Xi)
        tmp = 1 / (1 + expValue[0][0])
        if tmp >=0.5 and testLable[0][i] == 1 or tmp <0.5 and testLable[0][i] == 0:
            count_right += 1
    return count_right

##calculate the objective function
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
numOfIteration = 35
theta_list = []
loss_list = []
time_list = []
for i in range(0,numOfIteration):
    start = time.time()
    first_dir = first_derivative(trainData, trainDataLabel, theta, lamda)
    second_dir = second_derivative(trainData, theta, lamda)
    inversed_second_dir = np.linalg.inv(second_dir)
    
    thetaNew = theta - inversed_second_dir @ first_dir
    
#     count_right = testCase(test_data, test_label, thetaNew)
#     print(count_right / 1000)
    
    theta = thetaNew
    theta_list.append(theta)
    end = time.time()
    time_list.append(end - start)
    print(time_list[-1])
#     if i % 10 == 0:
    print(i)
    print(np.linalg.norm(theta))
    loss = obj_fun(trainData, trainDataLabel, theta, lamda)
    loss_list.append(loss)

# load the z* from the disk which we calculate individually
f = open("/Users/joechen/Desktop/545project/theta_start.txt", "r")
my_list = []

for line in f.readlines():
    my_list.append(float(line))

theta_start = np.array(my_list).reshape(len(my_list), 1)

# calculate the value for |z - z*|
res = []
for num in theta_list:
    res.append(np.linalg.norm(num - theta_start))
plt.plot(range(0, len(res)), res)

# calculate the ratio
res_ratio = []
for i in range(1, len(res)):
    res_ratio.append(res[i] / res[i - 1])
plt.title("SubNewton1")
plt.ylabel("rate")
plt.xlabel("Iterations")
plt.plot(range(0,len(res_ratio)), res_ratio)

