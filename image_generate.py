import vgg
import numpy as np
import tensorflow as tf
import cv2
import os
import wrapper
from functools import reduce
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def gram_matrix(matrix):
    return np.matmul(matrix.T,matrix)/matrix.size

content_layers = ('relu3_2','relu4_2', 'relu5_2')
style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
weight_path = "imagenet-vgg-verydeep-19.mat"
content_instance_path = 'content.jpg'
style_instance_path = 'style.jpg'
style_weight = 100000000000
total_variance_weight = 10000

vgg_weight,vgg_mean_pixel = vgg.load_net(weight_path)

content_instance = cv2.imread(content_instance_path)
content_shape = (1,) + content_instance.shape
content = tf.placeholder(shape=content_shape,dtype=tf.float32,name='content')

style_instance = cv2.imread(style_instance_path)
style_shape = (1,) + style_instance.shape
style = tf.placeholder(shape=style_shape,dtype=tf.float32,name='style')

content_feature = {}
style_feature = {}

@wrapper.run_time_record("Calculated content feature successfully!")
def calculate_content_feature():
    with tf.Session():
        VGG_net = vgg.net_preloaded(weights=vgg_weight, input_image=content, pooling='max')
        content_preprocess = np.array(content_instance) / 256
        for layer in content_layers:
            content_feature[layer] = VGG_net[layer].eval(feed_dict={content:np.reshape(content_preprocess,content_shape)})

@wrapper.run_time_record("Calculated style feature successfully!")
def calculate_style_feature():
    with tf.Session():
        VGG_net = vgg.net_preloaded(weights=vgg_weight,input_image=style,pooling='max')
        style_preprocess = np.array(style_instance) / 256
        for layer in style_layers:
            feature = VGG_net[layer].eval(feed_dict={style:np.reshape(style_preprocess,style_shape)})
            style_feature[layer] = gram_matrix(np.reshape(feature,[feature.shape[1]*feature.shape[2],feature.shape[3]]))

def generate_image(start_rate,decay_steps,end_step,export_period,beta1,beta2,epsilon):
    with tf.Graph().as_default() as g:
        image_shape = content_shape
        image = tf.Variable(tf.random_normal(mean=0, stddev=0.2, shape=image_shape), name='image',dtype=tf.float32)
        VGG_net = vgg.net_preloaded(weights=vgg_weight,input_image=image,pooling='max')
        content_loss = 0
        style_loss = 0
        alpha = np.full(len(content_feature),1/len(content_feature))
        beta = np.full(fill_value=1/len(style_feature),shape=len(style_feature))
        for i,layer in enumerate(content_layers):
            content_loss += alpha[i] * tf.losses.mean_squared_error(VGG_net[layer],content_feature[layer])
        for i,layer in enumerate(style_layers):
            current_layer = VGG_net[layer]
            feature = tf.reshape(current_layer,[current_layer.shape[1]*current_layer.shape[2],current_layer.shape[3]])
            feature = tf.matmul(tf.transpose(feature),feature)
            feature = feature / reduce(lambda x,y:x*y,current_layer.shape.as_list())
            style_loss += style_weight * beta[i] * tf.losses.mean_squared_error(feature,style_feature[layer])
        total_variance_loss = total_variance_weight * (tf.nn.l2_loss(image[:,1:,:,:]-image[:,:image_shape[1]-1,:,:]) + tf.nn.l2_loss(image[:,:,1:,:]-image[:,:,:image_shape[2]-1,:]))
        loss = content_loss + style_loss + total_variance_loss
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(start_rate, global_step, decay_steps=decay_steps, decay_rate=0.98,staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon).minimize(loss,global_step=global_step)

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())
        session.run(image.assign(np.reshape(content_instance,content_shape)/256))
        for i in range(end_step+1):
            if not i % export_period:
                image_instance = np.array(image.eval()*256)
                image_instance = np.reshape(np.array(np.clip(np.round(image_instance,0),0,255),dtype=int),newshape=[content_shape[1],content_shape[2],content_shape[3]])
                cv2.imwrite("step_"+str(i)+".jpg",image_instance,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            wrapper.run_time_record("step " + str(i) + ": loss " + str(loss.eval())
                                    + " |content loss " + str(content_loss.eval())
                                    + " |style loss " + str(style_loss.eval())
                                    + " |tv loss " + str(total_variance_loss.eval())
                                    + " |rate " + str(learning_rate.eval()))(train_step.run)(session=session)

calculate_content_feature()
calculate_style_feature()
generate_image(start_rate=0.007,decay_steps=1000,end_step=3000,export_period=2,beta1=0.9,beta2=0.999,epsilon=10e-8)



