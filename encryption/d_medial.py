import tensorflow as tf
import numpy as np,sys,os
from numpy import float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize


np.random.seed(678)
tf.set_random_seed(678)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),tf.float32)

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf.log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.asarray(dict[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict[b'labels'])
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[Yraw[i],i] = 1
    names = np.asarray(dict[b'filenames'])
    return X,Y,names

# make class 
class CNNLayer():
    
    def __init__(self,ker,in_c,out_c,act,d_act,):
        
        self.w = tf.Variable(tf.truncated_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,stride=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient,stride=1):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,filter_sizes = self.w.shape,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        grad_pass  = tf.nn.conv2d_backprop_input(
            input_sizes=[batch_size] + list(self.input.shape[1:]),filter = self.w ,
            out_backprop = grad_middle,strides=[1,1,1,1], padding="SAME"
        )

        update_w = []

        update_w.append(
            tf.assign( self.m,self.m*beta_1 + (1-beta_1) * grad   )
        )
        update_w.append(
            tf.assign( self.v,self.v*beta_2 + (1-beta_2) * grad ** 2   )
        )

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return grad_pass,update_w

data_location = "./big_image/"
data_array = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            data_array.append(os.path.join(dirName,filename))

X = np.zeros(shape=(100,128,128,1))
for file_index in range(len(data_array)):
    X[file_index,:,:]   = np.expand_dims(imresize(imread(data_array[file_index],mode='F',flatten=True),(128,128)),axis=3)

X[:,:,:,0] = (X[:,:,:,0]-X[:,:,:,0].min(axis=0))/(X[:,:,:,0].max(axis=0)-X[:,:,:,0].min(axis=0))

X = shuffle(X)
c_images = X[:70,:,:,:]

data_location = "./medical_image/"
data_array = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        data_array.append(os.path.join(dirName,filename))

X = np.zeros(shape=(70,128,128,1))
for file_index in range(len(data_array)):
    X[file_index,:,:]   = np.expand_dims(imresize(imread(data_array[file_index],mode='F',flatten=True),(128,128)),axis=3)
X[:,:,:,0] = (X[:,:,:,0]-X[:,:,:,0].min(axis=0))/(X[:,:,:,0].max(axis=0)-X[:,:,:,0].min(axis=0))
X = shuffle(X)
s_images = X

# hyper
num_epoch = 10000

learing_rate = 0.0001
batch_size = 10

networ_beta = 1.0

beta_1,beta_2 = 0.9,0.999
adam_e = 1e-8

# init class
prep_net1 = CNNLayer(3,1,50,tf_Relu,d_tf_Relu)
prep_net2 = CNNLayer(3,50,50,tf_Relu,d_tf_Relu)
prep_net3 = CNNLayer(3,50,50,tf_Relu,d_tf_Relu)
prep_net4 = CNNLayer(3,50,50,tf_Relu,d_tf_Relu)
prep_net5 = CNNLayer(3,50,1,tf_Relu,d_tf_Relu)

hide_net1 = CNNLayer(4,2,50,tf_Relu,d_tf_Relu)
hide_net2 = CNNLayer(4,50,50,tf_Relu,d_tf_Relu)
hide_net3 = CNNLayer(4,50,50,tf_Relu,d_tf_Relu)
hide_net4 = CNNLayer(4,50,50,tf_Relu,d_tf_Relu)
hide_net5 = CNNLayer(4,50,1,tf_Relu,d_tf_Relu)

reve_net1 = CNNLayer(5,1,50,tf_Relu,d_tf_Relu)
reve_net2 = CNNLayer(5,50,50,tf_Relu,d_tf_Relu)
reve_net3 = CNNLayer(5,50,50,tf_Relu,d_tf_Relu)
reve_net4 = CNNLayer(5,50,50,tf_Relu,d_tf_Relu)
reve_net5 = CNNLayer(5,50,1,tf_Relu,d_tf_Relu)

# make graph
Secret = tf.placeholder(shape=[None,128,128,1],dtype=tf.float32)
Cover = tf.placeholder(shape=[None,128,128,1],dtype=tf.float32)

prep_layer1 = prep_net1.feedforward(Secret)
prep_layer2 = prep_net2.feedforward(prep_layer1)
prep_layer3 = prep_net3.feedforward(prep_layer2)
prep_layer4 = prep_net4.feedforward(prep_layer3)
prep_layer5 = prep_net5.feedforward(prep_layer4)

hide_Input = tf.concat([Cover,prep_layer5],axis=3)
hide_layer1 = hide_net1.feedforward(hide_Input)
hide_layer2 = hide_net2.feedforward(hide_layer1)
hide_layer3 = hide_net3.feedforward(hide_layer2)
hide_layer4 = hide_net4.feedforward(hide_layer3)
hide_layer5 = hide_net5.feedforward(hide_layer4)

reve_layer1 = reve_net1.feedforward(hide_layer5)
reve_layer2 = reve_net2.feedforward(reve_layer1)
reve_layer3 = reve_net3.feedforward(reve_layer2)
reve_layer4 = reve_net4.feedforward(reve_layer3)
reve_layer5 = reve_net5.feedforward(reve_layer4)

cost_1 = tf.reduce_mean(tf.square(hide_layer5 - Cover))
cost_2 = tf.reduce_mean(tf.square(reve_layer5 - Secret)) 

# --- auto train ---
auto_train = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(cost_1+cost_2)


# start the session
with tf.Session() as sess : 

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        for current_batch_index in range(0,len(s_images),batch_size):
            current_batch_s = s_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_c = c_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost_1,cost_2,auto_train],feed_dict={Secret:current_batch_s,Cover:current_batch_c})
            print("Iter: ",iter, ' cost 1: ',sess_results[0],' cost 2: ',sess_results[1],end='\r')

        if iter % 50 == 0 :
            random_data_index = np.random.randint(len(s_images))
            current_batch_s = np.expand_dims(s_images[random_data_index,:,:,:],0)
            current_batch_c = np.expand_dims(c_images[random_data_index,:,:,:],0)
            sess_results = sess.run([prep_layer5,hide_layer5,reve_layer5],feed_dict={Secret:current_batch_s,Cover:current_batch_c})

            plt.figure()
            plt.imshow(np.squeeze(current_batch_s[0,:,:,:]),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' Secret')
            plt.savefig('images/epoch_'+str(iter)+"a_secret.png")

            plt.figure()
            plt.imshow(np.squeeze(current_batch_c[0,:,:,:]),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' cover')
            plt.savefig('images/epoch_'+str(iter)+"b_cover.png")

            plt.figure()
            plt.imshow(np.squeeze(sess_results[0][0,:,:,:]),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+' prep image')
            plt.savefig('images/epoch_'+str(iter)+"c_prep_images.png")

            plt.figure()
            plt.imshow(np.squeeze(sess_results[1][0,:,:,:]),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+" Hidden Image ")
            plt.savefig('images/epoch_'+str(iter)+"d_hidden_image.png")

            plt.figure()
            plt.axis('off')
            plt.imshow(np.squeeze(sess_results[2][0,:,:,:]),cmap='gray')
            plt.title('epoch_'+str(iter)+" Reveal  Image")
            plt.savefig('images/epoch_'+str(iter)+"e_reveal_images.png")

            plt.close('all')
            print('\n--------------------\n')

        if iter == num_epoch-1:
            
            for final in range(len(s_images)):
                current_batch_s = np.expand_dims(s_images[final,:,:,:],0)
                current_batch_c = np.expand_dims(c_images[final,:,:,:],0)
                sess_results = sess.run([prep_layer5,hide_layer5,reve_layer5],feed_dict={Secret:current_batch_s,Cover:current_batch_c})

                plt.figure()
                plt.imshow(np.squeeze(current_batch_s[0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title('epoch_'+str(final)+' Secret')
                plt.savefig('gif/epoch_'+str(final)+"a_secret.png")

                plt.figure()
                plt.imshow(np.squeeze(current_batch_c[0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title('epoch_'+str(final)+' cover')
                plt.savefig('gif/epoch_'+str(final)+"b_cover.png")

                plt.figure()
                plt.imshow(np.squeeze(sess_results[0][0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title('epoch_'+str(final)+' prep image')
                plt.savefig('gif/epoch_'+str(final)+"c_prep_images.png")

                plt.figure()
                plt.imshow(np.squeeze(sess_results[1][0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title('epoch_'+str(final)+" Hidden Image ")
                plt.savefig('gif/epoch_'+str(final)+"d_hidden_image.png")

                plt.figure()
                plt.axis('off')
                plt.imshow(np.squeeze(sess_results[2][0,:,:,:]),cmap='gray')
                plt.title('epoch_'+str(final)+" Reveal  Image")
                plt.savefig('gif/epoch_'+str(final)+"e_reveal_images.png")

                plt.close('all')
# -- end code --