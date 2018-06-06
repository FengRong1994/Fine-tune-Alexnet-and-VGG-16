import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  
import sklearn.metrics 
from sklearn.metrics.pairwise import cosine_similarity



def read_file(class_list):
    with open(class_list) as f:
        lines = f.readlines()
        images1 = []
        images2 = []
        labels = []
        for l in lines:
            items = l.split()
            images1.append(items[0])
            images2.append(items[1])
            labels.append(int(items[2]))
            
        #store total number of data
    return images1,images2, labels

#mean of imagenet dataset in BGR
imagenet_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)

image_dir = "train2.txt";

#%matplotlib inline

#get list of all images
img1,img2, labels = read_file(image_dir) 
#print("image")
#print(img_files)
#load all images

imgs1 = []
imgs2 = []
for i in img1:
    imgs1.append(cv2.imread(i))
for i in img2:
    imgs2.append(cv2.imread(i))
#print(imgs1[0])

from vgg import VGG16
from caffe_classes import class_names

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 224, 224, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = VGG16(x, keep_prob, 5749, ['fc8'])

#define activation of last layer as score
score = model.fc7
#f7=model.fc7;

saver=tf.train.Saver();
#create op to calculate softmax 
softmax=score
#softmax = tf.nn.softmax(score)
res=[]

with tf.Session() as sess:
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,"/finetune_vgg/model_epoch_16layers56.ckpt")
    # Load the pretrained weights into the model
    model.load_initial_weights(sess)
    
    
    fig2 = plt.figure(figsize=(15,6))
    
    sumc=0
    for count in range(1000):
        image1=cv2.resize(imgs1[count].astype(np.float32), (224,224))
        image2=cv2.resize(imgs2[count].astype(np.float32),(224,224))
        image1 -= imagenet_mean
        image2 -= imagenet_mean
        image1 = image1.reshape((1,224,224,3))
        image2 = image2.reshape((1,224,224,3))
        fea1=sess.run(softmax, feed_dict={x: image1, keep_prob: 1})
        
        fea2=sess.run(softmax,feed_dict={x: image2, keep_prob: 1})
        
        s=cosine_similarity(fea1,fea2)
        
        res.append(s[0,0])
       

fpr,tpr,threshold = sklearn.metrics.roc_curve(labels, res)   
roc_auc = sklearn.metrics.auc(fpr,tpr)
print(roc_auc)
 
lw = 2  

plt.plot(fpr, tpr, color='black41',  
         lw=lw)   
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('VGG16 with finetune')  
plt.legend(loc="lower right")  
plt.show()
 
