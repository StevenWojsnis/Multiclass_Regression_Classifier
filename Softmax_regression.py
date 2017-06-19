from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
from scipy import optimize as op
import numpy as np
import time
import sys


optimizationFlag = '1' # 0 means you want Gradient Descent, 1 means you want quasi-Newton
if(len(sys.argv) > 1 and (sys.argv[1] == '0' or sys.argv[1] == '1')):
    optimizationFlag = sys.argv[1]

dataset = raw_input("Enter 0 for iris, 1 for MNIST, or 2 for CIFAR-10: ")

# To unpack the cifar-10 file
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

iris = datasets.load_iris()
mnist = datasets.load_digits()

if(dataset == '2'):
    cifar = unpickle('/Users/stevewojsnis/Documents/Softmax_Regression/cifar-10-batches-py/data_batch_1')
    cifar2 = unpickle('/Users/stevewojsnis/Documents/Softmax_Regression/cifar-10-batches-py/data_batch_2')
    cifar3 = unpickle('/Users/stevewojsnis/Documents/Softmax_Regression/cifar-10-batches-py/data_batch_3')
    cifar4 = unpickle('/Users/stevewojsnis/Documents/Softmax_Regression/cifar-10-batches-py/data_batch_4')
    cifar5 = unpickle('/Users/stevewojsnis/Documents/Softmax_Regression/cifar-10-batches-py/data_batch_5')

    cifarData = np.concatenate([cifar['data'], cifar2['data'], cifar3['data'], cifar4['data'], cifar5['data']], axis=0)
    cifarLabels = np.concatenate([cifar['labels'], cifar2['labels'], cifar3['labels'], cifar4['labels'], cifar5['labels']], axis=0)

    cifarTest = unpickle('/Users/stevewojsnis/Documents/Softmax_Regression/cifar-10-batches-py/data_batch_2')
    cifarTestData = cifarTest['data']
    cifarTestLabels = cifarTest['labels']

# For Iris
if(dataset == '0'):
    index = range(0,149) # For comparison between Setosa and Versicolor

# For Mnist
if(dataset == '1'):
    index = range(0, mnist.data.shape[0]-1)

if(dataset == '2'):
    #index = range(0, cifar['data'].shape[0]-1)
    index = range(0, cifarData.shape[0]-1)

# Load Data
np.random.shuffle(index)

# For Iris
if(dataset == '0'):
    training_data = iris.data[index[:120]]
    training_target = iris.target[index[:120]]

    testing_data = iris.data[index[120:150]]
    testing_target = iris.target[index[120:150]]

# For Mnist
if(dataset == '1'):
    training_data = mnist.data[index[:mnist.data.shape[0]-300]]
    training_target = mnist.target[index[:mnist.target.shape[0]-300]]

    testing_data = mnist.data[index[mnist.data.shape[0]-300:mnist.data.shape[0]]]
    testing_target = mnist.target[index[mnist.target.shape[0]-300:mnist.target.shape[0]]]

if(dataset == '2'):

    training_data = cifarData[index[:]]
    training_target = cifarLabels[index[:]]

    testing_data = cifarTestData
    testing_target = cifarTestLabels

    pca = decomposition.PCA(n_components = 100);
    pca.fit(training_data)
    training_data = pca.transform(training_data)
    testing_data = pca.transform(testing_data)



#Normalizing doesn't work with the mnist dataset, the std of one sample is 0.
if(dataset != '1'):
    training_data = (training_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0) #Normalizing
    testing_data = (testing_data - np.mean(testing_data, axis=0)) / np.std(testing_data, axis=0) #Normalizing

# Adding ones to each of our samples, to allow for a bias.
training_data_ones = np.ones(training_data.shape[0])
training_data_ones = training_data_ones[:,np.newaxis]
training_data = np.concatenate((training_data,training_data_ones), axis = 1)

testing_data_ones = np.ones(testing_data.shape[0])
testing_data_ones = testing_data_ones[:,np.newaxis]
testing_data = np.concatenate((testing_data,testing_data_ones), axis = 1)

# Globally defining our losses list, to allow it to be accessed via BFGS, without having to pass it in as an argument
losses = []

#Utility Functions

#Note that, if using bfgs, have to expand w and flatten in upon return. This is
#because the fmin_bfgs function requires w to be a 1-D array.
def calcLoss(w,x,y):

    if(optimizationFlag == '1'):
        reshape_x = (w.shape[0])/training_data.shape[1]
        w = w.reshape([reshape_x, training_data.shape[1]])

    softmaxResult = softmax(w,x) #Number_Features x Number_classes
    logSoftmaxResult = np.log(softmaxResult)

    isolation = y * logSoftmaxResult

    loss = (-1/x.shape[0]) * np.sum(isolation) + (1/2)*np.sum(w*w)

    losses.append(loss)
    return loss

def softmax(w,x):
    alpha = x.dot(w.T)
    beta = np.exp(alpha)

    softmaxResult = beta.T / (np.sum(beta, axis = 1))
    return softmaxResult.T

#Note that, if using bfgs, have to expand w and flatten in upon return. This is
#because the fmin_bfgs function requires w to be a 1-D array.
def gradient(w,x,y):

    if(optimizationFlag == '1'):
        reshape_x = (w.shape[0])/training_data.shape[1]
        w = w.reshape([reshape_x, training_data.shape[1]])

    alpha = y - softmax(w,x)
    grad = alpha.T.dot(x) # This should be size k x N, where each row is a class, each col is a weight

    if(optimizationFlag == '1'):
        return ((-1/x.shape[0]) * grad + w).flatten()
    else:
        return ((-1/x.shape[0]) * grad + w)


def createOneHotMatrix(y):
    k = np.unique(y).shape[0] # Let k be the number of classes
    oneHotMat = np.zeros([y.shape[0], k])

    for rows in range(oneHotMat.shape[0]):
        oneHotMat[rows][y[rows]] = 1

    return oneHotMat

def classification(w,x):
    probabilities = softmax(w,x)
    predictions = np.argmax(probabilities, axis = 1)
    return probabilities,predictions

# Optimization Functions
def quasiNewton():
    w = np.zeros([np.unique(training_target).shape[0], training_data.shape[1]])
    st = time.time()
    Y = createOneHotMatrix(training_target)
    w = op.fmin_bfgs(calcLoss, w, fprime=gradient, epsilon=1e-5, args=(training_data,Y))
    et = time.time()

    print "First loss: ",losses[0],"  :  Final loss: ",losses[-1]
    print "Total optimization time: ",et-st

    reshape_x = (w.shape[0])/training_data.shape[1]
    w = w.reshape([reshape_x, training_data.shape[1]])

    return w

def gradientDescent():
    w = np.zeros([np.unique(training_target).shape[0], training_data.shape[1]])
    iterations = 1000
    learningRate = 1e-5

    Y = createOneHotMatrix(training_target)

    iterationTime = []

    for i in range(0,iterations):
        st = time.time()
        grad = gradient(w, training_data, Y)
        w = w - (learningRate * grad)

        loss = calcLoss(w, training_data, Y)

        if(losses and loss < losses[i-1]):
            learningRate = learningRate * 1.01
        else:
            learningRate = learningRate * .5

        et = time.time()
        iterationTime.append(et-st)

    print "First loss: ",losses[0],"  :  Final loss: ",losses[-1]
    print "Mean Iteration Time: ", np.mean(iterationTime)
    print "Number of Iterations: ", len(iterationTime)
    return w


if(optimizationFlag == '0'):
    w = gradientDescent()
else:
    w = quasiNewton()
probs, results = classification(w,testing_data)
print "Accuracy: ", float(np.sum(np.where(results == testing_target, 1, 0))) / results.shape[0]

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
