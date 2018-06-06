

import tensorflow as tf
import numpy as np

class AlexNet(object):
  
  def __init__(self, x, keep_prob, num_classes, skip_layer, 
               weights_path = 'DEFAULT'):
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer
    
    if weights_path == 'DEFAULT':      
      self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
    else:
      self.WEIGHTS_PATH = weights_path
    
    
    self.create()
    
  def create(self):
    
    
    conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
    norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
    
    
    conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
    norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
    
    # 3rd Layer: 
    conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')
    
    # 4th Layer: 
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')
    
    # 5th Layer: 
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')
    
    # 6th Layer: 
    flattened = tf.reshape(pool5, [-1, 6*6*256])
    fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
    dropout6 = dropout(fc6, self.KEEP_PROB)
    
    # 7th Layer: 
    self.fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
    self.dropout7 = dropout(self.fc7, self.KEEP_PROB)
    
    # 8th Layer: 
    self.fc8 = fc(self.dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')

    
    
  def load_initial_weights(self, session):
    """
  
    
   
            
     
  
"""
Predefine all necessary layer for the AlexNet
""" 

  """
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  