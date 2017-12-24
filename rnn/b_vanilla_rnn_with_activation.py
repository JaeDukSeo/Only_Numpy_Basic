import numpy as np

np.random.seed(1234)

def log(x):
    return 1 / ( 1+ np.exp( -1 * x ))

def d_log(x):
    return log(x) * (1 - log(x))

x = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,1]
])

y = np.array([
    [3],
    [2],
    [1]
])

wx = np.random.randn()
wrec = np.random.randn()
number_of_epoch = 15000

lr_wx = 0.001
lr_wrec = 0.001

state = np.zeros((x.shape[0],x.shape[1] + 1))
grad_over_time = np.zeros((x.shape))


for iter in range(number_of_epoch):

    state_1_in  = state[:,0]*wrec + x[:,0]*wx
    state_1_out = log(state_1_in)
    state[:,1] = state_1_out

    state_2_in  = state[:,1]*wrec + x[:,1]*wx
    state_2_out = log(state_2_in)
    state[:,2] = state_2_out

    state_3_in  = state[:,2]*wrec + x[:,2]*wx
    state[:,3] = state_3_in

    cost = np.square(state[:,3] - np.squeeze(y)).sum() / len(x)

    if iter % 1000 == 0:
        print("Current iter : ", iter, " Current cost: ", cost)

    grad_over_time[:,2] = (state[:,3] - np.squeeze(y)) * (2/len(x))
    grad_over_time[:,1] = grad_over_time[:,2] * wrec  * d_log(state_2_in)
    grad_over_time[:,0] = grad_over_time[:,1] * wrec  * d_log(state_1_in)

    grad_wx = np.sum(grad_over_time[:,2]*x[:,2]+
                    grad_over_time[:,1]*x[:,1]+
                    grad_over_time[:,0]*x[:,0])

    grad_wrec = np.sum(grad_over_time[:,2]*state[:,2]+
                    grad_over_time[:,1]*state[:,1]+
                    grad_over_time[:,0]*state[:,0])

    wx = wx - lr_wx * grad_wx
    wrec = wrec - lr_wrec * grad_wrec
    
# 3. Final Output
state_1_in  = state[:,0]*wrec + x[:,0]*wx
state_1_out = log(state_1_in)
state[:,1] = state_1_out

state_2_in  = state[:,1]*wrec + x[:,1]*wx
state_2_out = log(state_2_in)
state[:,2] = state_2_out

state_3_in  = state[:,2]*wrec + x[:,2]*wx
state[:,3] = state_3_in
    
print("-----------\n")    
print("Final output Raw: ",state_3_in)
print("Final output Rounded: ",np.round(state_3_in))
print("Ground Truth : ",y.T)
print("-----------\n")    


    
    
#  --- end code ---
