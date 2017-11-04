import numpy as np

# ----- ASSUME -------
# 1. pyhon - comfortable
# 2. NN - understanding 

# ------ NETWORK ARCH ----
# 1. Epoch 100
# 2. LR = 1
# 3. SGD - BATCH 
# 4. Loss - MSE with added 1/2


# Generate Same Random Numbers Every Time
np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp( -1 * x))

def d_sigmoid(x):
    return sigmoid(x) * (1- sigmoid(x))

# 0. Data Preprocess and etc...
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [1]
])

# 0.5 Declare Hyper Parameter
w1 = np.random.randn(3,1)
numer_of_epoch =100
learning_rate = 10

# 1. Make the Opertions
for iter in range(numer_of_epoch):

    # 1. Make the Dot Product operation
    layer_1 = x.dot(w1)
    layer_1_act = sigmoid(layer_1)

    # loss function - MSE 0.5
    loss = np.square(layer_1_act - y) / (len(layer_1_act)  * 2)

    print "Current Epoch : ",iter ," Current Accuracy : ",1- loss.sum()," current loss :", loss.sum()," Current Learning Rate: ",learning_rate

    # SGD - BATCH
    grad_1_part_1 = (layer_1_act - y) / len(layer_1_act)
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = x
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    # Weight Update
    w1 -=  learning_rate * grad_1
    
    if iter == 50 :
        learning_rate = 1
    if iter == 70 :
        learning_rate = 0.1

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

print "\n\nFinal : " ,layer_1_act[:,-1]
print "Final Round: " ,np.round(layer_1_act[:,-1])
print "Ground Truth : ",y[:,-1]
print "W1 : ",w1[:,-1]




# ------ END CODE -------