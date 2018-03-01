import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt,os,sys
from sklearn.utils import shuffle
from scipy.ndimage import imread
from numpy import float32
# -2. Set the Random Seed Values
tf.set_random_seed(789)
np.random.seed(568)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth=True

# -1 Tf activation functions
def tf_arctan(x):
    return tf.atan(x)
def d_tf_arctan(x):
    return 1.0/(1+tf.square(x))

def tf_tanh(x):
    return tf.tanh(x)
def d_tf_tanh(x):
    return 1. - tf.square(tf_tanh(x))

def tf_ReLU(x):
    return tf.nn.relu(x)
def d_tf_ReLu(x):
    return tf.cast(tf.greater(x, 0),dtype=tf.float32)

def tf_log(x):
    return tf.sigmoid(x)
def d_tf_log(x):
    return tf.sigmoid(x) * (1.0 - tf.sigmoid(x))

# 0. Get the list
PathDicom = "../../lung_data_1/"
lstFilesDCM1 = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(PathDicom)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM1.append(os.path.join(dirName,filename))
PathDicom = "../../lung_data_2/"
lstFilesDCM2 = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(PathDicom)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM2.append(os.path.join(dirName,filename))
PathDicom = "../../lung_data_3/"
lstFilesDCM3 = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(PathDicom)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM3.append(os.path.join(dirName,filename))

# 1. Read the data into Numpy
one = np.zeros((119,512,512))
two = np.zeros((119,512,512))
three = np.zeros((119,512,512))

# 1.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM1)):
    one[file_index,:,:]   = imread(lstFilesDCM1[file_index],mode='F')
    two[file_index,:,:]   = imread(lstFilesDCM2[file_index],mode='F')
    three[file_index,:,:]   = imread(lstFilesDCM3[file_index],mode='F')
print('===== Done READING DATA ========')

# 1.75 Make the Training data and make it fir 
training_data = np.vstack((one,two,three[:-2,:,:]))
path = 'dialted3/'
if not os.path.exists(path):
    os.makedirs(path)

# Make Hyper Parameter
num_epoch = 101
batch_size = 5
learning_rate = 0.000000001

proportion_rate = 800
decay_rate = 0.064

beta1,beta2 = 0.9,0.999
adam_e = 0.00000001

# Make Class
class ResCNNLayer():
    
    def __init__(self,kernel_size=None,channel_in=None,channel_out=None,act=None,d_act=None):
        
        self.w = tf.Variable(tf.random_normal([kernel_size,kernel_size,channel_in,channel_out]))

        self.input,self.output = None,None
        self.act,self.d_act = act,d_act

        self.layer = None
        self.layerA = None
        self.layerRes = None

        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feed_forward(self,input=None,og_input=None):
        self.input = input
        self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerA =  self.act(self.layer)
        self.layerRes = self.output = tf.add(tf.multiply(self.layerA,og_input),og_input)
        return self.output

    def backprop(self,gradient=None,og_input=None):
        grad_part_1 = tf.multiply(gradient, og_input)
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.transpose(tf.transpose(tf.multiply(grad_part_1,grad_part_2),[0,2,1,3]),[0,2,1,3])

        grad = tf.nn.conv2d_backprop_filter(input=grad_part_3,filter_sizes=self.w.shape,
        out_backprop=grad_middle,strides=[1,1,1,1],padding='SAME')

        pass_size = list(self.input.shape[1:])
        pass_on_grad = tf.nn.conv2d_backprop_input(input_sizes=[batch_size]+pass_size,filter=self.w,
        out_backprop=grad_middle,strides=[1,1,1,1],padding='SAME')

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))
        # grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(learning_rate,grad))))
        

        return pass_on_grad,grad_update

    def getw(self):
        return self.w

# Make the Netwokr
l1 = ResCNNLayer(13,1,3,tf_log,d_tf_log)
l2 = ResCNNLayer(7,3,3,tf_tanh,d_tf_tanh)
l3 = ResCNNLayer(3,3,3,tf_arctan,d_tf_arctan)

l4 = ResCNNLayer(5,3,3,tf_ReLU,d_tf_ReLu)
l5 = ResCNNLayer(3,3,3,tf_tanh,d_tf_tanh)
l6 = ResCNNLayer(1,3,1,tf_ReLU,d_tf_ReLu)

l1w,l2w,l3w,l4w,l5w,l6w = l1.getw(),l2.getw(),l3.getw(),l4.getw(),l5.getw(),l6.getw()

# Make the graph
x = tf.placeholder(shape=[None,512,512,1],dtype="float")
y = tf.placeholder(shape=[None,512,512,1],dtype="float")

iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

layer1 = l1.feed_forward(x,x)
layer2 = l2.feed_forward(layer1,x)
layer3 = l3.feed_forward(layer2,x)

layer4 = l4.feed_forward(layer3,x)
layer5 = l5.feed_forward(layer4,x)
layer6 = l6.feed_forward(layer5,x)

loss = tf.reduce_sum(tf.square(tf.subtract(layer6,y) * 0.5))

grad_6,g6w = l6.backprop(tf.subtract(layer6,y),x)
grad_5,g5w = l5.backprop(grad_6,x)
grad_4,g4w = l4.backprop(grad_5+decay_propotoin_rate*(grad_6),x)

grad_3,g3w = l3.backprop(grad_4+decay_propotoin_rate*(grad_6*grad_5),x)
grad_2,g2w = l2.backprop(grad_3+decay_propotoin_rate*(grad_6*grad_5*grad_4),x)
grad_1,g1w = l1.backprop(grad_2+decay_propotoin_rate*(grad_6*grad_5*grad_4*grad_3),x)

update = g1w+g2w+g3w+g4w+g5w+g6w

# Make the Session
total_cost = 0 
with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    # Traing for Epoch
    for iter in range(num_epoch):
        
        for current_batch_index in range(0,len(training_data),batch_size):

            current_batch = training_data[current_batch_index:current_batch_index+batch_size,:,:]
            current_batch_noise =  current_batch + 0.3 * current_batch.max() *np.random.randn(current_batch.shape[0],current_batch.shape[1],current_batch.shape[2])

            current_batch       = float32(np.expand_dims(current_batch,axis=3)) 
            current_batch_noise = float32(np.expand_dims(current_batch_noise,axis=3))

            auto_results = sess.run([loss,update],feed_dict={x:current_batch_noise,y:current_batch,iter_variable_dil:iter})
            print("Current Iter: ", iter," Current batch : ",current_batch_index ," Current Loss: ",auto_results[0],end='\r')
            total_cost = total_cost + auto_results[0]
        if iter%5==0: 
            print("\nCurrent Iter: ", iter," Total Cost until now: ",total_cost,'\n')
        total_cost = 0

        if iter%10==0:
            # After All oc the num epoch training make sample output
            for image_in_one in range(0,len(one)):
                
                current_image = np.expand_dims(one[image_in_one,:,:],axis=0)
                current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])
                current_image      = float32(np.expand_dims(current_image,axis=3)) 
                current_data_noise = float32(np.expand_dims(current_data_noise,axis=3))
                temp = sess.run(layer6,feed_dict={x:current_data_noise})

                f, axarr = plt.subplots(2, 2)
                axarr[0, 0].imshow(np.squeeze(current_image[0,:,:,:]),cmap='gray')
                axarr[0, 0].set_title('Original Image at :' + str(image_in_one))

                axarr[0, 1].imshow(np.squeeze(current_data_noise[0,:,:,:]),cmap='gray')
                axarr[0, 1].set_title('Noise Image at :' + str(image_in_one))
                
                axarr[1, 0].imshow(np.squeeze(temp[0,:,:,:]),cmap='gray')
                axarr[1, 0].set_title('Denoise Image at :' + str(image_in_one))

                plt.savefig(path+str(image_in_one)+'.png')
                plt.close('all')

    # After All oc the num epoch training make sample output
    for image_in_one in range(0,len(one)):
        
        current_image = np.expand_dims(one[image_in_one,:,:],axis=0)
        current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])
        current_image      = float32(np.expand_dims(current_image,axis=3)) 
        current_data_noise = float32(np.expand_dims(current_data_noise,axis=3))
        temp = sess.run(layer6,feed_dict={x:current_data_noise})

        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(np.squeeze(current_image[0,:,:,:]),cmap='gray')
        axarr[0, 0].set_title('Original Image at :' + str(image_in_one))

        axarr[0, 1].imshow(np.squeeze(current_data_noise[0,:,:,:]),cmap='gray')
        axarr[0, 1].set_title('Noise Image at :' + str(image_in_one))
        
        axarr[1, 0].imshow(np.squeeze(temp[0,:,:,:]),cmap='gray')
        axarr[1, 0].set_title('Denoise Image at :' + str(image_in_one))

        plt.savefig(path+str(image_in_one)+'.png')
        plt.close('all')


# -- end code --