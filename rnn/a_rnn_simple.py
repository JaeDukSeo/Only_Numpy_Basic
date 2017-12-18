import numpy as np

np.random.seed(1234)

# 1. Data Preprocess and declare 
x = np.array([
    [1,0,0],
    [1,1,0],
    [1,1,1]
])

y = np.array([
    [1],
    [2],
    [3]
])

wx = [0.2]
wrec = [1.5]

number_or_epoch = 100
number_of_training_data = 3
learning_rate_x = 0.02
learning_rate_rec = 0.0006

states = np.zeros((3,4))
grad_over_time = np.zeros((3,4))

# 2. Start the Training
for iter in range(number_or_epoch):

    # 2.3 Feed Forward of the network
    layer_1 = x[:,0] * wx + states[:,0] * wrec
    states[:,1] = layer_1

    layer_2 = x[:,1] * wx + states[:,1] * wrec
    states[:,2] = layer_2

    layer_3 = x[:,2] * wx + states[:,2] * wrec
    states[:,3] = layer_3

    cost = np.square(states[:,3] - y).sum() / number_of_training_data

    grad_out = (states[:,3] - np.squeeze(y)) * 2 / number_of_training_data
    grad_over_time[:,3] = grad_out
    grad_over_time[:,2] = grad_over_time[:,3] * wrec
    grad_over_time[:,1] = grad_over_time[:,2] * wrec

    # NOTE: Do Not really need grad_over_time[:,0]
    grad_over_time[:,0] = grad_over_time[:,1] * wrec

    grad_wx = np.sum(grad_over_time[:,3] * x[:,2] + grad_over_time[:,2] * x[:,1]  + grad_over_time[:,1] * x[:,0])
    grad_rec = np.sum(grad_over_time[:,3] * states[:,2] + grad_over_time[:,2] * states[:,1]  + grad_over_time[:,1] * states[:,0])
    
    wx = wx - learning_rate_x * grad_wx
    wrec = wrec - learning_rate_rec * grad_rec

    print('Current Epoch: ',iter, '  current predition :' ,layer_3)
    

# 3. Final Output and rounded resutls
layer_1 = x[:,0] * wx + states[:,0] * wrec
states[:,1] = layer_1

layer_2 = x[:,1] * wx + states[:,1] * wrec
states[:,2] = layer_2

layer_3 = x[:,2] * wx + states[:,2] * wrec
states[:,3] = layer_3

print('Ground Truth: ',layer_3)
print('Rounded Truth: ',np.round(layer_3))




# ---- end code ---