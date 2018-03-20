import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

np.random.seed(68)
tf.set_random_seed(5678)

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_softmax(x): return tf.nn.softmax(x)

# Function to unpcicle
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

PathDicom = "../z_CIFAR_data/cifar10batchespy/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if not ".html" in filename.lower() and not  ".meta" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

global train_batch,train_label,test_batch,test_label

# Read the data traind and Test
batch0 = unpickle(lstFilesDCM[0])
batch1 = unpickle(lstFilesDCM[1])
batch2 = unpickle(lstFilesDCM[2])
batch3 = unpickle(lstFilesDCM[3])
batch4 = unpickle(lstFilesDCM[4])

onehot_encoder = OneHotEncoder(sparse=True)

train_batch = np.vstack((batch0[b'data'],batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data']))
train_label = np.expand_dims(np.hstack((batch0[b'labels'],batch1[b'labels'],batch2[b'labels'],batch3[b'labels'],batch4[b'labels'])).T,axis=1).astype(np.float32)
train_label = onehot_encoder.fit_transform(train_label).toarray().astype(np.float32)

test_batch = unpickle(lstFilesDCM[5])[b'data']
test_label = np.expand_dims(np.array(unpickle(lstFilesDCM[5])[b'labels']),axis=0).T.astype(np.float32)
test_label = onehot_encoder.fit_transform(test_label).toarray().astype(np.float32)

# Normalize data from 0 to 1
train_batch = (train_batch - train_batch.min(axis=0))/(train_batch.max(axis=0)-train_batch.min(axis=0))
test_batch = (test_batch - test_batch.min(axis=0))/(test_batch.max(axis=0)-test_batch.min(axis=0))

# reshape data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# rotate data
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)

# cnn
class CNNLayer():
    
    def __init__(self,kernel,inchan,outchan):
        self.w = tf.Variable(tf.random_normal([kernel,kernel,inchan,outchan]))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
    def getw(self): return self.w
    def feedforward(self,input,stride_num=1):
        self.input = input
        self.layer = tf.nn.conv2d(input,self.w,strides=[1,stride_num,stride_num,1],padding="SAME")
        self.layerA = tf_Relu(self.layer)
        return self.layerA
    
    def backprop(self,gradient,stride_num=1):
        
        grad_part_1 = gradient
        grad_part_2 = d_tf_Relu(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,
            out_backprop = grad_middle,
            strides = [1,stride_num,stride_num,1],
            padding='SAME'
        )

        pass_on_grad = tf.nn.conv2d_backprop_input(
            input_sizes= [batch_size, self.input.shape[1].value, self.input.shape[2].value, self.input.shape[3].value],
            filter = self.w,
            out_backprop = grad_middle,
            strides = [1,stride_num,stride_num,1],
            padding="SAME"
        )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return pass_on_grad,grad_update             


# fcc
class FCCLayer():
    
    def __init__(self,input,output):
        self.w = tf.Variable(tf.random_normal([input,output]))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
    def getw(self): return self.w
    def feedforward(self,input):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = tf_log(self.layer)
        return self.layerA

    def backprop(self,gradient):
        
        grad_part_1 = gradient
        grad_part_2 = d_tf_log(self.layer)
        grad_part_3 = self.input

        grad = tf.matmul(tf.transpose(grad_part_3),tf.multiply(grad_part_1,grad_part_2))
        pass_on_grad = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return pass_on_grad,grad_update             

# Hyper Param
learning_rate = 0.001
batch_size = 200
num_epoch = 800
print_size = 50

beta1,beta2 = 0.9,0.999
adam_e = 0.00000001

proportion_rate = 1000
decay_rate = 0.008

def case0_auto_method():
    
    global train_batch,train_label,test_batch,test_label
    tf.reset_default_graph()

    # create layers
    l1 = CNNLayer(5,3,64)
    l2 = CNNLayer(5,64,64)
    l3 = CNNLayer(5,64,64)
    l4 = CNNLayer(3,64,64)
    l5 = CNNLayer(3,64,64)

    l6 = FCCLayer(4096,1024)
    l7 = FCCLayer(1024,1024)
    l8 = FCCLayer(1024,1024)
    l9 = FCCLayer(1024,10)

    weights = [l1.getw(),l2.getw(),
            l3.getw(),l4.getw(),
            l5.getw(),l6.getw(),
            l7.getw(),l8.getw(),
            l9.getw()]

    # Create graph
    x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
    y = tf.placeholder(shape=[None,10],dtype=tf.float32)

    iter_variable_dil = tf.placeholder(tf.float32, shape=())
    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

    layer1 = l1.feedforward(x)
    layer2 = l2.feedforward(layer1)
    layer3 = l3.feedforward(layer2)

    layer4 = l4.feedforward(layer3,stride_num=2)
    layer5 = l5.feedforward(layer4,stride_num=2)

    layer6_Input = tf.reshape(layer5,[batch_size,-1])
    layer6 = l6.feedforward(layer6_Input)
    layer7 = l7.feedforward(layer6)
    layer8 = l8.feedforward(layer7)
    layer9 = l9.feedforward(layer8)

    final_soft = tf_softmax(layer9)

    cost = tf.reduce_sum(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
    correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # auto train
    auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=weights)

    # create Session
    with tf.Session(config=config) as sess: 

        sess.run(tf.global_variables_initializer())

        train_total_cost,train_total_acc =0,0
        train_cost_overtime,train_acc_overtime = [],[]

        test_total_cost,test_total_acc = 0,0
        test_cost_overtime,test_acc_overtime = [],[]

        for iter in range(num_epoch):
            
            train_batch,train_label = shuffle(train_batch,train_label)

            # Train Batch
            for current_batch_index in range(0,len(train_batch),batch_size):

                current_batch = train_batch[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_batch_label = train_label[current_batch_index:current_batch_index+batch_size,:]

                sess_results = sess.run( [cost,accuracy,correct_prediction,auto_train], feed_dict= {x:current_batch,y:current_batch_label,iter_variable_dil:iter})
                print("current iter:", iter, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
                train_total_cost = train_total_cost + sess_results[0]
                train_total_acc = train_total_acc + sess_results[1]

            # Test batch
            for current_batch_index in range(0,len(test_batch),batch_size):

                current_batch = test_batch[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_batch_label = test_label[current_batch_index:current_batch_index+batch_size,:]

                sess_results = sess.run( [cost,accuracy,correct_prediction], feed_dict= {x:current_batch,y:current_batch_label,iter_variable_dil:iter})
                print("current iter:", iter, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
                test_total_cost = test_total_cost + sess_results[0]
                test_total_acc = test_total_acc + sess_results[1]

            # store
            train_cost_overtime.append(train_total_cost/(len(train_batch)/batch_size ) )
            train_acc_overtime.append(train_total_acc/(len(train_batch)/batch_size ) )

            test_cost_overtime.append(test_total_cost/(len(test_batch)/batch_size ) )
            test_acc_overtime.append(test_total_acc/(len(test_batch)/batch_size ) )
            
            # print
            if iter%print_size == 0:
                print('\n=========')
                print("Avg Train Cost: ", train_cost_overtime[-1])
                print("Avg Train Acc: ", train_acc_overtime[-1])
                print("Avg Test Cost: ", test_cost_overtime[-1])
                print("Avg Test Acc: ", test_acc_overtime[-1])
                print('-----------')      
            train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0            
                  
            
    return train_cost_overtime,train_acc_overtime,test_cost_overtime,test_acc_overtime

def case1_Dilated():
    global train_batch,train_label,test_batch,test_label
    
    tf.reset_default_graph()

    # create layers
    l1 = CNNLayer(5,3,64)
    l2 = CNNLayer(5,64,64)
    l3 = CNNLayer(5,64,64)
    l4 = CNNLayer(3,64,64)
    l5 = CNNLayer(3,64,64)

    l6 = FCCLayer(4096,1024)
    l7 = FCCLayer(1024,1024)
    l8 = FCCLayer(1024,1024)
    l9 = FCCLayer(1024,10)

    weights = [l1.getw(),l2.getw(),
            l3.getw(),l4.getw(),
            l5.getw(),l6.getw(),
            l7.getw(),l8.getw(),
            l9.getw()]

    # Create graph
    x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
    y = tf.placeholder(shape=[None,10],dtype=tf.float32)

    iter_variable_dil = tf.placeholder(tf.float32, shape=())
    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

    layer1 = l1.feedforward(x)
    layer2 = l2.feedforward(layer1)
    layer3 = l3.feedforward(layer2)

    layer4 = l4.feedforward(layer3,stride_num=2)
    layer5 = l5.feedforward(layer4,stride_num=2)

    layer6_Input = tf.reshape(layer5,[batch_size,-1])
    layer6 = l6.feedforward(layer6_Input)
    layer7 = l7.feedforward(layer6)
    layer8 = l8.feedforward(layer7)
    layer9 = l9.feedforward(layer8)

    final_soft = tf_softmax(layer9)

    cost = tf.reduce_sum(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
    correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # back propagation
    grad_9,grad_9w = l9.backprop(final_soft-y)
    grad_8,grad_8w = l8.backprop(grad_9)
    grad_7,grad_7w = l7.backprop(grad_8+decay_propotoin_rate*(grad_9))
    grad_6,grad_6w = l6.backprop(grad_7+decay_propotoin_rate*(grad_9+grad_8))

    grad_5_Input = tf.reshape(grad_6,[batch_size,8,8,64])
    grad_5,grad_5w = l5.backprop(grad_5_Input,2)
    grad_4,grad_4w = l4.backprop(grad_5,2)

    grad_3,grad_3w = l3.backprop(grad_4)
    grad_2,grad_2w = l2.backprop(grad_3+decay_propotoin_rate*(grad_4))
    grad_1,grad_1w = l1.backprop(grad_2+decay_propotoin_rate*(grad_4+grad_3))

    grad_update = grad_9w+grad_8w+grad_7w+grad_6w+\
                grad_5w+grad_4w+grad_3w+grad_2w+grad_1w

    # auto train
    auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=weights)

    # create Session
    with tf.Session(config=config) as sess: 

        sess.run(tf.global_variables_initializer())

        train_total_cost,train_total_acc =0,0
        train_cost_overtime,train_acc_overtime = [],[]

        test_total_cost,test_total_acc = 0,0
        test_cost_overtime,test_acc_overtime = [],[]

        for iter in range(num_epoch):
            
            train_batch,train_label = shuffle(train_batch,train_label)

            # Train Batch
            for current_batch_index in range(0,len(train_batch),batch_size):

                current_batch = train_batch[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_batch_label = train_label[current_batch_index:current_batch_index+batch_size,:]

                sess_results = sess.run( [cost,accuracy,correct_prediction,grad_update], feed_dict= {x:current_batch,y:current_batch_label,iter_variable_dil:iter})
                print("current iter:", iter, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
                train_total_cost = train_total_cost + sess_results[0]
                train_total_acc = train_total_acc + sess_results[1]

            # Test batch
            for current_batch_index in range(0,len(test_batch),batch_size):

                current_batch = test_batch[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_batch_label = test_label[current_batch_index:current_batch_index+batch_size,:]

                sess_results = sess.run( [cost,accuracy,correct_prediction], feed_dict= {x:current_batch,y:current_batch_label,iter_variable_dil:iter})
                print("current iter:", iter, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
                test_total_cost = test_total_cost + sess_results[0]
                test_total_acc = test_total_acc + sess_results[1]

            # store
            train_cost_overtime.append(train_total_cost/(len(train_batch)/batch_size ) )
            train_acc_overtime.append(train_total_acc/(len(train_batch)/batch_size ) )

            test_cost_overtime.append(test_total_cost/(len(test_batch)/batch_size ) )
            test_acc_overtime.append(test_total_acc/(len(test_batch)/batch_size ) )
            
            # print
            if iter%print_size == 0:
                print('\n=========')
                print("Avg Train Cost: ", train_cost_overtime[-1])
                print("Avg Train Acc: ", train_acc_overtime[-1])
                print("Avg Test Cost: ", test_cost_overtime[-1])
                print("Avg Test Acc: ", test_acc_overtime[-1])
                print('-----------')    
            train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0            
                    
            
    return train_cost_overtime,train_acc_overtime,test_cost_overtime,test_acc_overtime

def case2_Dilated2():
    
    global train_batch,train_label,test_batch,test_label
    tf.reset_default_graph()

    # create layers
    l1 = CNNLayer(5,3,64)
    l2 = CNNLayer(5,64,64)
    l3 = CNNLayer(5,64,64)
    l4 = CNNLayer(3,64,64)
    l5 = CNNLayer(3,64,64)

    l6 = FCCLayer(4096,1024)
    l7 = FCCLayer(1024,1024)
    l8 = FCCLayer(1024,1024)
    l9 = FCCLayer(1024,10)

    weights = [l1.getw(),l2.getw(),
            l3.getw(),l4.getw(),
            l5.getw(),l6.getw(),
            l7.getw(),l8.getw(),
            l9.getw()]

    # Create graph
    x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
    y = tf.placeholder(shape=[None,10],dtype=tf.float32)

    iter_variable_dil = tf.placeholder(tf.float32, shape=())
    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

    layer1 = l1.feedforward(x)
    layer2 = l2.feedforward(layer1)
    layer3 = l3.feedforward(layer2)

    layer4 = l4.feedforward(layer3,stride_num=2)
    layer5 = l5.feedforward(layer4,stride_num=2)

    layer6_Input = tf.reshape(layer5,[batch_size,-1])
    layer6 = l6.feedforward(layer6_Input)
    layer7 = l7.feedforward(layer6)
    layer8 = l8.feedforward(layer7)
    layer9 = l9.feedforward(layer8)

    final_soft = tf_softmax(layer9)

    cost = tf.reduce_sum(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
    correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # back propagation
    grad_9,grad_9w = l9.backprop(final_soft-y)
    grad_8,grad_8w = l8.backprop(grad_9+decay_propotoin_rate*(grad_9))
    grad_7,grad_7w = l7.backprop(grad_8+decay_propotoin_rate*(grad_9+grad_8))
    grad_6,grad_6w = l6.backprop(grad_7+decay_propotoin_rate*(grad_9+grad_8+grad_7))

    grad_5_Input = tf.reshape(grad_6,[batch_size,8,8,64])
    grad_5,grad_5w = l5.backprop(grad_5_Input+decay_propotoin_rate*(grad_5_Input),2)
    grad_4,grad_4w = l4.backprop(grad_5+decay_propotoin_rate*(grad_5),2)

    grad_3,grad_3w = l3.backprop(grad_4+decay_propotoin_rate*(grad_4))
    grad_2,grad_2w = l2.backprop(grad_3+decay_propotoin_rate*(grad_4+grad_3))
    grad_1,grad_1w = l1.backprop(grad_2+decay_propotoin_rate*(grad_4+grad_3+grad_2))

    grad_update = grad_9w+grad_8w+grad_7w+grad_6w+\
                grad_5w+grad_4w+grad_3w+grad_2w+grad_1w

    # create Session
    with tf.Session(config=config) as sess: 

        sess.run(tf.global_variables_initializer())

        train_total_cost,train_total_acc =0,0
        train_cost_overtime,train_acc_overtime = [],[]

        test_total_cost,test_total_acc = 0,0
        test_cost_overtime,test_acc_overtime = [],[]

        for iter in range(num_epoch):
            
            train_batch,train_label = shuffle(train_batch,train_label)

            # Train Batch
            for current_batch_index in range(0,len(train_batch),batch_size):

                current_batch = train_batch[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_batch_label = train_label[current_batch_index:current_batch_index+batch_size,:]

                sess_results = sess.run( [cost,accuracy,correct_prediction,grad_update], feed_dict= {x:current_batch,y:current_batch_label,iter_variable_dil:iter})
                print("current iter:", iter, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
                train_total_cost = train_total_cost + sess_results[0]
                train_total_acc = train_total_acc + sess_results[1]

            # Test batch
            for current_batch_index in range(0,len(test_batch),batch_size):

                current_batch = test_batch[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_batch_label = test_label[current_batch_index:current_batch_index+batch_size,:]

                sess_results = sess.run( [cost,accuracy,correct_prediction], feed_dict= {x:current_batch,y:current_batch_label,iter_variable_dil:iter})
                print("current iter:", iter, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
                test_total_cost = test_total_cost + sess_results[0]
                test_total_acc = test_total_acc + sess_results[1]

            # store
            train_cost_overtime.append(train_total_cost/(len(train_batch)/batch_size ) )
            train_acc_overtime.append(train_total_acc/(len(train_batch)/batch_size ) )

            test_cost_overtime.append(test_total_cost/(len(test_batch)/batch_size ) )
            test_acc_overtime.append(test_total_acc/(len(test_batch)/batch_size ) )
            
            # print
            if iter%print_size == 0:
                print('\n=========')
                print("Avg Train Cost: ", train_cost_overtime[-1])
                print("Avg Train Acc: ", train_acc_overtime[-1])
                print("Avg Test Cost: ", test_cost_overtime[-1])
                print("Avg Test Acc: ", test_acc_overtime[-1])
                print('-----------')  
            train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0            
            
    return train_cost_overtime,train_acc_overtime,test_cost_overtime,test_acc_overtime

# results
# train_cost_overtime00,train_acc_overtime00,test_cost_overtime00,test_acc_overtime00 = case_original()
train_cost_overtime0,train_acc_overtime0,test_cost_overtime0,test_acc_overtime0 = case0_auto_method()
train_cost_overtime1,train_acc_overtime1,test_cost_overtime1,test_acc_overtime1 = case1_Dilated()
train_cost_overtime2,train_acc_overtime2,test_cost_overtime2,test_acc_overtime2 = case2_Dilated2()

# plot and save
plt.figure()
# plt.plot(range(len(train_cost_overtime0)),train_cost_overtime00,color='y',label='Original Model')
plt.plot(range(len(train_cost_overtime0)),train_cost_overtime0,color='r',label='Modified Auto')
plt.plot(range(len(train_cost_overtime0)),train_cost_overtime1,color='g',label='Modified Dilated 1')
plt.plot(range(len(train_cost_overtime0)),train_cost_overtime2,color='b',label='Modified Dilated 2')
plt.legend()
plt.savefig('Train Cost over time')

plt.figure()
# plt.plot(range(len(train_cost_overtime0)),train_acc_overtime00,color='y',label='Original Model')
plt.plot(range(len(train_acc_overtime0)),train_acc_overtime0,color='r',label='Modified Auto')
plt.plot(range(len(train_acc_overtime0)),train_acc_overtime1,color='g',label='Modified Dilated 1')
plt.plot(range(len(train_acc_overtime0)),train_acc_overtime2,color='b',label='Modified Dilated 2')
plt.legend()
plt.savefig('Train Acc over time')

plt.figure()
# plt.plot(range(len(test_cost_overtime0)),test_cost_overtime00,color='y',label='Original Model')
plt.plot(range(len(test_cost_overtime0)),test_cost_overtime0,color='r',label='Modified Auto')
plt.plot(range(len(test_cost_overtime0)),test_cost_overtime1,color='g',label='Modified Dilated 1')
plt.plot(range(len(test_cost_overtime0)),test_cost_overtime2,color='b',label='Modified Dilated 2')
plt.legend()
plt.savefig('Test Cost over time')

plt.figure()
# plt.plot(range(len(test_acc_overtime0)),test_acc_overtime00,color='y',label='Original Model')
plt.plot(range(len(test_acc_overtime0)),test_acc_overtime0,color='r',label='Modified Auto')
plt.plot(range(len(test_acc_overtime0)),test_acc_overtime1,color='g',label='Modified Dilated 1')
plt.plot(range(len(test_acc_overtime0)),test_acc_overtime2,color='b',label='Modified Dilated 2')
plt.legend()
plt.savefig('Test Acc over time')

# --- end code --