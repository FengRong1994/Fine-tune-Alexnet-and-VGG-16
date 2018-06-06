
import tensorflow as tf
import numpy as np

class VGG16(object):
  
  def __init__(self, x, keep_prob, num_classes, skip_layer, 
               weights_path = 'DEFAULT'):
    
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer
    
    if weights_path == 'DEFAULT':      
      self.WEIGHTS_PATH = 'vgg16.npy'
    else:
      self.WEIGHTS_PATH = weights_path
    
    self.create()
    
  def create(self):
  
        conv1_1 = conv(self.X, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
        conv1_2 = conv(conv1_1    , 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
        pool1   = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        
        # 2nd Layer: Conv -> Conv -> Pool
        conv2_1 = conv(pool1  , 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
        pool2   = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # 3rd Layer: Conv -> Conv -> Conv -> Pool
        conv3_1 = conv(pool2  , 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
        pool3   = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # 4th Layer: Conv -> Conv -> Conv -> Pool
        conv4_1 = conv(pool3  , 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
        pool4   = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # 5th Layer: Conv -> Conv -> Conv -> Pool
        conv5_1 = conv(pool4  , 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
        pool5   = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        # 6th Layer: FC -> DropOut
        # [1:] cuts away the first element
        pool5_out  = int(np.prod(pool5.get_shape()[1:])) # 7 * 7 * 512 = 25088
        pool5_flat = tf.reshape(pool5, [-1, pool5_out]) 
        # shape=(image count, 7, 7, 512) -> shape=(image count, 25088)
        fc6        = fc(pool5_flat,25088 ,num_out=4096, name='fc6', relu=True)
        dropout1   = tf.nn.dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC
        self.fc7      = fc(dropout1,4096, num_out=4096, name='fc7', relu=True)
        self.dropout2 = tf.nn.dropout(self.fc7, self.KEEP_PROB)

        # 8th Layer: FC
        self.fc8 = fc(self.dropout2,4096, num_out=self.NUM_CLASSES, name='fc8', relu=False)
    
    
  def load_initial_weights(self, session):
    
    weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
    
    for op_name in weights_dict:
      op_name_string = op_name if isinstance(op_name, str) else op_name.decode('utf8')
      
      if op_name_string not in self.SKIP_LAYER:
        
        with tf.variable_scope(op_name_string, reuse = True):
            
          
          for data in weights_dict[op_name]:
            
            # Biases
            if len(data.shape) == 1:
              
              var = tf.get_variable('biases', trainable = False)
              session.run(var.assign(data))
              
            # Weights
            else:
              
              var = tf.get_variable('weights', trainable = False)
              session.run(var.assign(data))
            
     
  

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
  
  input_channels = int(x.get_shape()[-1])
  
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name) as scope:
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    
    if groups == 1:
      conv = convolve(x, weights)
    else:
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      conv = tf.concat(axis = 3, values = output_groups)
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
  
    relu = tf.nn.relu(bias, name = scope.name)
        
    return relu
  
def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)
  
def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
  
def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)
  
    