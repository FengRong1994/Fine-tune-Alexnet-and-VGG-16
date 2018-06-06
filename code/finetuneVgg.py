
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from vgg import VGG16
from datagenerator import ImageDataGenerator

# Path to the textfiles for the trainings and validation set
train_file = 'train.txt'
val_file = 'val.txt'

learning_rate = 0.002
num_epochs = 45
batch_size = 16

dropout_rate = 0.5
num_classes = 5749

train_layers = ['conv5_3','fc6','fc7','fc8']

display_step = 128

filewriter_path = "/finetune_alexnet/dogs_vs_cats"
checkpoint_path = "/finetune_alexnet/"

if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

model = VGG16(x, keep_prob, num_classes, train_layers)

score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

with tf.name_scope("train"):
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)
 
for var in var_list:
  tf.summary.histogram(var.name, var)
  
tf.summary.scalar('cross_entropy', loss)
  

with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
tf.summary.scalar('accuracy', accuracy)


merged_summary = tf.summary.merge_all()


writer = tf.summary.FileWriter(filewriter_path)


saver = tf.train.Saver()


train_generator = ImageDataGenerator(train_file, 
                                     horizontal_flip = True, shuffle = True, mean = np.array([127.5, 127.5, 127.5]), scale_size=(224, 224),
                 nb_classes = 5749)
val_generator = ImageDataGenerator(val_file, shuffle = False) 

train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)


with tf.Session() as sess:
 
  # Initialize all variables
  #sess.run(tf.global_variables_initializer())
  saver.restore(sess,"D:/finetune_alexnet/model_epoch_164layers34.ckpt")
  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)
  
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        while step < train_batches_per_epoch:
          
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, 
                                          y: batch_ys, 
                                          keep_prob: dropout_rate})
            
            # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                        y: batch_ys, 
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                
            step += 1
        
        train_generator.reset_pointer()
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(train_batches_per_epoch):
            batch_tx, batch_ty = train_generator.next_batch(batch_size)
            
            acc = sess.run(accuracy, feed_dict={x: batch_tx, 
                                                y: batch_ty, 
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
            print(test_acc)
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
        
        
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now())) 
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch_1643layers'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        if (test_acc >= 0.95):
          break
