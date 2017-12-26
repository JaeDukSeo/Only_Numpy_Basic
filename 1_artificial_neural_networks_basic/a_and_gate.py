import numpy as np

np.random.seed(1234)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - tanh(x) ** 2

x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([
    [0],
    [0],
    [0],
    [1]
])

w1 = np.random.randn(3,5)
w2 = np.random.randn(5,1)

for iter in range(300):

    layer_1 = x.dot(w1)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = tanh(layer_2)

    cost = np.square(layer_2_act - y).sum() / len(x)

    grad_2_part_1 = (2/len(x)) * (layer_2_act - y)
    grad_2_part_2 = d_tanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =   grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2) 

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = x
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)   

    w1 -= 0.01*grad_1
    w2 -= 0.1*grad_2
    


layer_1 = x.dot(w1)
layer_1_act = tanh(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = tanh(layer_2)
print(layer_2_act)


# -- end code --