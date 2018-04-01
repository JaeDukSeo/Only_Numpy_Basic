import os
import numpy
from matplotlib import pyplot as plt, cm
import io,gzip,zipfile
import numpy as np,sys,os
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
plt.style.use('ggplot')
tf.set_random_seed(789)
np.random.seed(568)

# -1 Tf activation functions
def tf_arctan(x): return tf.atan(x)
def d_tf_arctan(x): return 1.0/(1+tf.square(x))

# def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLu(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)

def tf_tanh(x):return tf.tanh(x)
def d_tf_tanh(x):return 1. - tf.square(tf_tanh(x))

def tf_log(x):return tf.sigmoid(x)
def d_tf_log(x):return tf.sigmoid(x) * (1.0 - tf.sigmoid(x))

def tf_ReLU(x): return tf.nn.elu(x)
def d_tf_elu(x): 
  mask1 = tf.cast(tf.greater(x,0),dtype=tf.flaot32)
  maks2 = tf_elu(tf.cast(tf.less_equal(x,0),dtype=tf.float32) * x)
  return mask1 + mask2

def tf_softmax(x): return tf.nn.softmax(x)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# ---- make class ----
class ConLayer():
  
  def __init__(self,kernel,in_c,out_c,act,d_act):
    self.w  = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c], stddev=0.005))
    self.act,self.d_act = act,d_act
    self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

  def feedforward(self,input,stride=1):
    self.input = input
    self.layer  = tf.contrib.layers.batch_norm(tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding='SAME'))
    self.layerA = self.act(self.layer)
    return self.layerA

  def backpropagation(self,gradient,stride=1):
    grad_part1 = gradient
    grad_part2 = self.d_act(self.layer)
    grad_part3 = self.input

    grad_middle = tf.multiply(grad_part1,grad_part2)

    grad = tf.nn.conv2d_backprop_filter(
      input = grad_part3,
      filter_sizes = self.w.shape,
      out_backprop = grad_middle,
      strides=[1,stride,stride,1],padding='SAME'
    )

    grad_pass = tf.nn.conv2d_backprop_input(
      input_sizes = [batch_size] + list(self.input.shape[1:]),
      filter = self.w,
      out_backprop = grad_middle,
      strides=[1,stride,stride,1],padding='SAME'
    )

    update_w = []

    update_w.append( tf.assign( self.m,self.m * beta1 + (1-beta1) * grad     )   )
    update_w.append( tf.assign( self.v,self.v * beta2 + (1-beta2) * grad  ** 2   )   )

    m_hat = self.m/(1-beta1)
    v_hat = self.v/(1-beta2)
    adam_middle = init_lr / ( tf.sqrt(v_hat) + adam_e)

    update_w.append(
      tf.assign(self.w, self.w - adam_middle * m_hat   )
    )

    return grad_pass,update_w

class fnnlayer():
    def __init__(self,input_dim,hidden_dim,activation,d_activation):
        self.w = tf.Variable(tf.truncated_normal([input_dim,hidden_dim], stddev=0.005))
        self.act,self.d_act  = activation,d_activation
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feed_forward(self,input=None):
        self.input = input
        self.layer = tf.contrib.layers.batch_norm(tf.matmul(input,self.w))
        self.layerA = self.act(self.layer)
        return self.layerA

    def backpropagation(self,gradient=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input 

        grad_x_mid = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.matmul(tf.transpose(grad_part_3),grad_x_mid)

        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        update_w = []

        update_w.append( tf.assign( self.m,self.m * beta1 + (1-beta1) * grad     )   )
        update_w.append( tf.assign( self.v,self.v * beta2 + (1-beta2) * grad  ** 2   )   )

        m_hat = self.m/(1-beta1)
        v_hat = self.v/(1-beta2)
        adam_middle = init_lr / ( tf.sqrt(v_hat) + adam_e)
        update_w.append(
          tf.assign(self.w, self.w - adam_middle * m_hat   )
        )
        return grad_pass,update_w



# --- get data ---
PathDicom = "./data/cifar-10-batches-py/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if not ".html" in filename.lower() and not  ".meta" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

batch0 = unpickle(lstFilesDCM[0])
batch1 = unpickle(lstFilesDCM[1])
batch2 = unpickle(lstFilesDCM[2])
batch3 = unpickle(lstFilesDCM[3])
batch4 = unpickle(lstFilesDCM[4])
onehot_encoder = OneHotEncoder(sparse=True)

train_batch = np.vstack((batch0[b'data'],batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data']))
train_label = np.expand_dims(np.hstack((batch0[b'labels'],batch1[b'labels'],batch2[b'labels'],batch3[b'labels'],batch4[b'labels'])).T,axis=1).astype(np.float32)
train_labels = onehot_encoder.fit_transform(train_label).toarray().astype(np.float32)

test_batch = unpickle(lstFilesDCM[5])[b'data']
test_label = np.expand_dims(np.array(unpickle(lstFilesDCM[5])[b'labels']),axis=0).T.astype(np.float32)
test_labels = onehot_encoder.fit_transform(test_label).toarray().astype(np.float32)

# reshape data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# rotate data
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)

# Normalize data from 0 to 1
train_batch[:,:,:,0] = (train_batch[:,:,:,0] - train_batch[:,:,:,0].min(axis=0))/(train_batch[:,:,:,0].max(axis=0)-train_batch[:,:,:,0].min(axis=0))
train_batch[:,:,:,1] = (train_batch[:,:,:,1] - train_batch[:,:,:,1].min(axis=0))/(train_batch[:,:,:,1].max(axis=0)-train_batch[:,:,:,1].min(axis=0))
train_batch[:,:,:,2] = (train_batch[:,:,:,2] - train_batch[:,:,:,2].min(axis=0))/(train_batch[:,:,:,2].max(axis=0)-train_batch[:,:,:,2].min(axis=0))

test_batch[:,:,:,0] = (test_batch[:,:,:,0] - test_batch[:,:,:,0].min(axis=0))/(test_batch[:,:,:,0].max(axis=0)-test_batch[:,:,:,0].min(axis=0))
test_batch[:,:,:,1] = (test_batch[:,:,:,1] - test_batch[:,:,:,1].min(axis=0))/(test_batch[:,:,:,1].max(axis=0)-test_batch[:,:,:,1].min(axis=0))
test_batch[:,:,:,2] = (test_batch[:,:,:,2] - test_batch[:,:,:,2].min(axis=0))/(test_batch[:,:,:,2].max(axis=0)-test_batch[:,:,:,2].min(axis=0))

train_images = train_batch
test_images  = test_batch

# === Hyper Parameter ===
num_epoch =  100
batch_size = 100
print_size = 1
shuffle_size = 1
divide_size = 10

init_lr = 0.001

proportion_rate = 1000
decay_rate = 0.008
# decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

beta1,beta2 = 0.9,0.999
adam_e = 0.00000001

one_channel = 84
one_vector  = 1645

# === make classes ====
l1_1 = ConLayer(3,3,one_channel,tf_ReLU,d_tf_ReLu)
l1_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l1_s = ConLayer(1,3,one_channel,tf_ReLU,d_tf_ReLu)

l1_1_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l1_2_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l1_s_2 = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l2_1_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_1_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_1_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l2_2_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_2_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_2_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l2_3_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_3_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_3_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l3_1_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_1_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_1_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l3_2_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_2_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_2_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l3_3_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_3_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_3_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l4_Input_shape = 8*8* (one_channel *3)
l4_1 = fnnlayer(l4_Input_shape,one_vector,tf_ReLU,d_tf_ReLu)
l4_2 = fnnlayer(one_vector,one_vector,tf_ReLU,d_tf_ReLu)
l4_s = fnnlayer(l4_Input_shape,one_vector,tf_ReLU,d_tf_ReLu)

l5_1 = fnnlayer(one_vector,one_vector,tf_ReLU,d_tf_ReLu)
l5_2 = fnnlayer(one_vector,10,tf_ReLU,d_tf_ReLu)
l5_s = fnnlayer(one_vector,10,tf_ReLU,d_tf_ReLu)

# --- make graph ----
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)
learing_rate_train= tf.placeholder(tf.float32, shape=())

layer1_1 = l1_1.feedforward(x,stride=2)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_s = l1_s.feedforward(x,stride=2)
layer1_add = layer1_s + layer1_2

layer1_1_2 = l1_1_2.feedforward(layer1_add,stride=2)
layer1_2_2 = l1_2_2.feedforward(layer1_1_2)
layer1_s_2 = l1_s_2.feedforward(layer1_add,stride=2)
layer1_add_2 = layer1_s_2 + layer1_2_2

# --- node layer 2 -----
layer2_1_1 = l2_1_1.feedforward(layer1_add_2)
layer2_1_2 = l2_1_2.feedforward(layer2_1_1)
layer2_1_s = l2_1_s.feedforward(layer1_add_2)
layer2_1_add = layer2_1_s + layer2_1_2

layer2_2_1 = l2_2_1.feedforward(layer1_add_2)
layer2_2_2 = l2_2_2.feedforward(layer2_1_1)
layer2_2_s = l2_2_s.feedforward(layer1_add_2)
layer2_2_add = layer2_2_s + layer2_2_2

layer2_3_1 = l2_3_1.feedforward(layer1_add_2)
layer2_3_2 = l2_3_2.feedforward(layer2_3_1)
layer2_3_s = l2_3_s.feedforward(layer1_add_2)
layer2_3_add = layer2_3_s + layer2_3_2

# --- node layer 3 -----
layer3_Input = layer2_1_add + layer2_2_add + layer2_3_add
layer3_1_1 = l3_1_1.feedforward(layer3_Input)
layer3_1_2 = l3_1_2.feedforward(layer3_1_1)
layer3_1_s = l3_1_s.feedforward(layer3_Input)
layer3_1_add = layer3_1_s + layer3_1_2

layer3_2_1 = l3_2_1.feedforward(layer3_Input)
layer3_2_2 = l3_2_2.feedforward(layer3_2_1)
layer3_2_s = l3_2_s.feedforward(layer3_Input)
layer3_2_add = layer3_2_s + layer3_2_2

layer3_3_1 = l3_3_1.feedforward(layer3_Input)
layer3_3_2 = l3_3_2.feedforward(layer3_3_1)
layer3_3_s = l3_3_s.feedforward(layer3_Input)
layer3_3_add = layer3_3_s + layer3_3_2

# ---- fully connected layer ----
layer3_a = layer3_1_add + layer3_2_add
layer3_b = layer3_2_add + layer3_3_add
layer3_c = layer3_1_add + layer3_3_add

layer4_Input = tf.reshape(tf.concat([layer3_a,layer3_b,layer3_c],axis=3),[batch_size,-1])
layer4_1 = l4_1.feed_forward(layer4_Input)
layer4_2 = l4_2.feed_forward(layer4_1)
layer4_s = l4_s.feed_forward(layer4_Input)
layer4_add = layer4_s+layer4_2

layer5_1 = l5_1.feed_forward(layer4_add)
layer5_2 = l5_2.feed_forward(layer5_1)
layer5_s = l5_s.feed_forward(layer4_add)
layer5_add = layer5_s+layer5_2

# --- final layer ---
final_soft = tf_softmax(layer5_add)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer5_add,labels=y)) * 0.5
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# -- auto train --
auto_train = tf.train.AdamOptimizer(learning_rate=learing_rate_train).minimize(cost)


# === Start the Session ===
with tf.Session() as sess: 

    # start the session 
    sess.run(tf.global_variables_initializer())
    train_total_cost,train_total_acc, test_total_cost,test_total_acc =0,0,0,0
    train_cost_overtime,train_acc_overtime,test_cost_overtime,test_acc_overtime = [],[],[],[]

    # Start the Epoch
    for iter in range(num_epoch):
        
        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy,auto_train,correct_prediction,final_soft], feed_dict={x: current_batch, y: current_batch_label,iter_variable_dil:iter,learing_rate_train:init_lr})
            print("current iter:", iter,' learning rate: %.6f'%init_lr ,
                ' Current batach : ',current_batch_index," current cost: %.38f" % sess_results[0],' current acc: %.5f '%sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
            current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy],feed_dict={x: current_batch, y: current_batch_label})
            print("Test Image Current iter:", iter,' learning rate: %.6f'%init_lr,
                 ' Current batach : ',current_batch_index, " current cost: %.38f" % sess_results[0],' current acc: %.5f '%sess_results[1], end='\r')
            test_total_cost = test_total_cost + sess_results[0]
            test_total_acc = test_total_acc + sess_results[1]

        # store
        train_cost_overtime.append(train_total_cost/(len(train_images)/divide_size/batch_size )  ) 
        train_acc_overtime.append(train_total_acc /(len(train_images)/divide_size/batch_size )  )
        test_cost_overtime.append(test_total_cost/(len(test_images)/batch_size ))
        test_acc_overtime.append(test_total_acc/(len(test_images)/batch_size ))
            
        # print
        if iter%print_size == 0:
            print('\n\n==== Current Iter :', iter,' Average Results =====')
            print("Avg Train Cost: %.18f"% train_cost_overtime[-1])
            print("Avg Train Acc:  %.18f"% train_acc_overtime[-1])
            print("Avg Test Cost:  %.18f"% test_cost_overtime[-1])
            print("Avg Test Acc:   %.18f"% test_acc_overtime[-1])
            print('=================================')      

        # shuffle 
        if iter%shuffle_size ==  0: 
            print("==== shuffling iter: ",iter," =======\n")
            train_images,train_labels = shuffle(train_images,train_labels)

        # redeclare
        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0

        # real time ploting
        if iter > 0: plt.clf()
        plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train COT")
        plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test COT')
        plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='g',label="Train AOT")
        plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='y',label='Test AOT')
        plt.legend()
        plt.axis('auto')
        plt.title('Results')
        plt.pause(0.1)

    # plot and save
    plt.clf()
    plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train COT")
    plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test COT')
    plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='g',label="Train AOT")
    plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='y',label='Test AOT')
    plt.legend()
    plt.title('Results')
    plt.show()





# -- end code ---