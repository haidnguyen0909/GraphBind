# -*- coding: utf-8 -*-
"""Deepbind.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ElLcihYoUY4RqhGQQmOMIH1kHdtmSH1f
"""

# Load the Drive helper and mount
#from google.colab import drive

# This will prompt for authorization.
#drive.mount('/content/drive')
# After executing the cell above, Drive
# files will be present in "/content/drive/My Drive".
#!ls "/content/drive/My Drive"

#!ls "/content/drive/My Drive/Colab Notebooks/"

import tensorflow as tf
import numpy as np
import csv
import math 
import random
import gzip
from scipy.stats import bernoulli

nummotif=300 #number of motifs to discover
bases='ACGT' #DNA bases
basesRNA='ACGU'#RNA bases
batch_size=1000 #fixed batch size -> see notes to problem about it
dictReverse={'A':'T','C':'G','G':'C','T':'A','N':'N'} #dictionary to implement reverse-complement mode
class Experiment:
    def __init__(self,filename,motiflen):
        self.file=filename
        self.motiflen=motiflen
    
    def getMotifLen(self):
        return self.motiflen

def seqtopad(sequence,motlen,kind='DNA'):
    rows=len(sequence)+2*motlen-2
    S=np.empty([rows,4])
    base= bases if kind=='DNA' else basesRNA
    for i in range(rows):
        for j in range(4):
            if i-motlen+1<len(sequence) and sequence[i-motlen+1]=='N' or i<motlen-1 or i>len(sequence)+motlen-2:
                S[i,j]=np.float32(0.25)
            elif sequence[i-motlen+1]==base[j]:
                S[i,j]=np.float32(1)
            else:
                S[i,j]=np.float32(0)
    return S

def dinucshuffle(sequence):
    b=[sequence[i:i+2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d=''.join([str(x) for x in b])
    return d

def logsampler(a,b):
    x=tf.Variable(tf.random_uniform([],minval=0,maxval=1), trainable=False)
    y=10**((math.log10(b)-math.log10(a))*x + math.log10(a))
    
#     x=np.random.uniform(low=0,high=1)
#     y=10**((math.log10(b)-math.log10(a))*x + math.log10(a))
    return y

def sqrtsampler(a,b):
    x=tf.Variable(tf.random_uniform([],minval=0,maxval=1), trainable=False)
#     x=np.random.uniform(low=0,high=1)
    y=(b-a)*(x**0.5)+a
    return y

def modifystr(str, size):
  n = len(str)
  return str[size:n-size]
SIZE = 0
class Chip(Experiment):
    def __init__(self,filename,motiflen=24):
        self.file = filename
        self.motiflen = motiflen
            
    def openFile(self):
        train_dataset=[]
        count=0
        header =True
        with open(self.file) as data:
            reader = data.readlines()
            
            for row in reader:
              if header:
                header=False
                continue
              row=row.rstrip().split('\t')
              seq = row[2]
              train_dataset.append([seqtopad(seq,self.motiflen),[1]])
              train_dataset.append([seqtopad(dinucshuffle(seq),self.motiflen),[0]])
              count+=1
              if count >= 500:
                break
                   
        
        
        #random.shuffle(train_dataset)
        #print(train_dataset)
        #exit(1)

        frac1=int(len(train_dataset)*1/3)
        frac2=int(len(train_dataset)*2/3)
        return train_dataset[:frac1],train_dataset[frac1:frac2],train_dataset[frac2:],train_dataset

#filename='./data/encode/ARID3A_K562_ARID3A_(sc-8821)_Stanford_B.seq'

filename = "./data/encode/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_B.seq"
class ChipTest(Experiment):
    def __init__(self,filename,motiflen=24):
        self.file = filename
        self.motiflen = motiflen
            
    def openFile(self):
        train_dataset=[]
        count = 0
        header = True

        with open(self.file) as data:
            reader = data.readlines()
            
            for row in reader:
              if header:
                header=False
                continue
              row = row.rstrip().split('\t')
              seq = row[2]
              train_dataset.append([seqtopad(row[2],self.motiflen),[int(row[3])]])
                    
                   
        return train_dataset

import time




filename = "./data/encode/BATF_GM12878_BATF_HudsonAlpha_AC.seq"
test= Chip(filename)
d1,d2,d3,dataAll =test.openFile()

filename = "./data/encode/BATF_GM12878_BATF_HudsonAlpha_B.seq"
test= ChipTest(filename)
data_test =test.openFile()

data1=np.asarray([el[0] for el in d1],dtype=np.float32)
label1=np.asarray([el[1] for el in d1],dtype=np.float32).reshape(len(data1),1)

data2=np.asarray([el[0] for el in d2],dtype=np.float32)
label2=np.asarray([el[1] for el in d2],dtype=np.float32).reshape(len(data2),1)

data3=np.asarray([el[0] for el in d3],dtype=np.float32)
label3=np.asarray([el[1] for el in d3],dtype=np.float32).reshape(len(data3),1)

data=[data1,data2,data3]
label=[label1,label2,label3]

data_all=np.asarray([el[0] for el in dataAll],dtype=np.float32)
label_all=np.asarray([el[1] for el in dataAll],dtype=np.float32).reshape(len(data_all),1)

d_test=np.asarray([el[0] for el in data_test],dtype=np.float32)
l_test=np.asarray([el[1] for el in data_test],dtype=np.float32).reshape(len(data_test),1)


def convolution(input_data, num_input_channels, num_filters, filter_shape, conv_weights,bias_weights,wd1,bd1,W,b,pooling,neuType,training,dropprob):

    
    # setup the convolutional layer operation
    out_layer = tf.nn.conv1d(input_data, conv_weights, 1, padding='VALID')

    out_layer= tf.subtract(out_layer,conv_bias)

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform pooling
    if pooling == 'max_pool':
        pool=tf.reduce_max(out_layer,axis=1) 
        
    elif pooling == 'avg_pool':
        out_layer1= tf.reduce_max(out_layer, axis=1)
        out_layer2= tf.reduce_mean(out_layer, axis=1)
        
        x_expanded = tf.expand_dims(out_layer1, 2)                 
        y_expanded = tf.expand_dims(out_layer2, 2)  
        
        concatted = tf.concat([x_expanded, y_expanded], 2)  

        pool = tf.reshape(concatted, [-1, 2*num_filters]) 
        

    t =tf.constant(1 ,dtype=tf.float32)
    
    def ifTrain(pool):
        pooldrop = tf.nn.dropout(pool,keep_prob=dropprob)
#         pooldrop=tf.multiply(pool,mask) 
        out = tf.matmul(pooldrop, wd1) + bd1
        
        return out
    def ifTest(pool):
        out = dropprob*tf.matmul(pool, wd1) + bd1
        return out
    
    #check if there's hidden stage
    if(neuType=='nohidden'):

        out = tf.cond(tf.equal(training,t), lambda: ifTrain(pool), lambda: ifTest(pool))

        
    elif(neuType=='hidden'):


        dense_layer1 = tf.matmul(pool, W) + b
        dense_layer1=tf.nn.relu(dense_layer1)
        
        out = tf.cond(tf.equal(training,t), lambda: ifTrain(dense_layer1), lambda: ifTest(dense_layer1))
        

    return out

graph=tf.Graph()
with graph.as_default():
    
    num_input_channels=4
    num_filters=nummotif
    filter_shape=10
    pooling='max_pool'
    neuType='nohidden'
    
    beta1=tf.placeholder_with_default(logsampler(10**-15,10**-3),shape=())
    beta2=tf.placeholder_with_default(logsampler(10**-10,10**-3),shape=())
    beta3=tf.placeholder_with_default(logsampler(10**-10,10**-3),shape=())

    
    
    learning_rate= tf.placeholder_with_default(logsampler(0.0005, 0.05),shape=())
    momentum_rate= tf.placeholder_with_default(sqrtsampler(0.95, 0.99),shape=())
    



    batch_size=1000
    with tf.device('/cpu:0'):
    
      x = tf.placeholder(tf.float32, [None, 147, 4])
      y = tf.placeholder(tf.float32,[None,1])
      dropprob = tf.placeholder_with_default(1.0, shape=())

      # Distinguish training and testing: training=1 for training , =0 for testing
      training = tf.placeholder_with_default(0.0, shape=())

    with tf.device('/cpu:0'):
      
      #Set up iterator for the data
      dataset = tf.data.Dataset.from_tensor_slices((x, y))
      dataset = dataset.shuffle(500).repeat().batch(batch_size)
      
                  
      iterator = dataset.make_initializable_iterator()
      data_X, data_y = iterator.get_next()      
      data_y = tf.cast(data_y, tf.float32)
      
      
   
    with tf.device('/cpu:0'):
      
      conv_filt_shape = [filter_shape, num_input_channels, num_filters]

      stdConv=tf.placeholder_with_default(logsampler(10**-7,10**-3),shape=()) 
      # initialise weights and bias for the filter
      conv_weights = tf.Variable(tf.truncated_normal(conv_filt_shape, mean=0,stddev=stdConv), name='Conv1_W')
      conv_bias = tf.Variable(tf.truncated_normal([num_filters]), name='Conv1_b')



      if pooling=='max_pool':
          W = tf.Variable(tf.truncated_normal([16,32], mean=0, stddev=0.3), name='W')
          b = tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.3), name='b')
      else:
          W = tf.Variable(tf.truncated_normal([32,32], mean=0, stddev=0.3), name='W')
          b = tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.3), name='b')     

      if neuType == 'nohidden':
          if pooling=='max_pool':
              wdim1=nummotif
          else:
              wdim1=2 * nummotif
      else:
          wdim1= 2 * nummotif
          
      stdNeu=tf.placeholder_with_default(logsampler(10**-5,10**-2) ,shape=()) 
      wd1 = tf.Variable(tf.truncated_normal([wdim1,1], mean=0, stddev=stdNeu), name='w2')
      bd1 = tf.Variable(tf.truncated_normal([1], mean=0, stddev=stdNeu), name='b2')


      xconv = convolution(data_X,num_input_channels,num_filters,filter_shape,conv_weights,conv_bias,wd1,bd1,W,b,pooling,neuType,training,dropprob)


      sig = tf.nn.sigmoid(xconv)
      if neuType == 'hidden':
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_y,logits=xconv))+ beta1*tf.norm(conv_weights,ord=1)+ beta2*tf.norm(wd1,ord=1)+ beta3*tf.norm(W,ord=1)
      else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_y,logits=xconv))+ beta1*tf.norm(conv_weights,ord=1)+ beta2*tf.norm(wd1,ord=1)

      optimizer=tf.train.MomentumOptimizer(learning_rate,momentum_rate,use_nesterov=True).minimize(loss)

    with tf.device('/cpu:0'):
      #Set up iterator for the validation data
      dataset_val = tf.data.Dataset.from_tensor_slices((x, y))
      dataset_val = dataset_val.batch(tf.cast(tf.size(y),tf.int64))
                  
      iterator_val = dataset_val.make_initializable_iterator()
      data_XV, data_yV = iterator_val.get_next()      
      data_yV = tf.cast(data_yV, tf.float32)
    with tf.device('/cpu:0'):
      xconvV = convolution(data_XV,num_input_channels,num_filters,filter_shape,conv_weights,conv_bias,wd1,bd1,W,b,pooling,neuType,training,dropprob)

      sigV = tf.nn.sigmoid(xconvV)

import copy
from sklearn import metrics
import numpy as np
import random

with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False)) as sess:
    dropoutList=[0.5] #list of possible dropout values
    best_AUC=0

    for iter in range(10):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        beta_1=sess.run(beta1)
        beta_2=sess.run(beta2)
        beta_3=sess.run(beta3)
        lea_r,mom_r,stdc,stdn=sess.run([learning_rate,momentum_rate,stdConv,stdNeu])
                
        
        prob=random.choice(dropoutList)
        
        crossV=[0,1,2]
    
        CV_auc_list=[]
        Avg_List=[]
        for c in [0]:
              start_time = time.time()
              sess.run(tf.global_variables_initializer())
              sess.run(tf.local_variables_initializer())
              sess.run([conv_weights,wd1,conv_bias,bd1], feed_dict={stdConv:stdc,stdNeu:stdn})
              t=copy.copy(crossV)
              #t.remove(c)
              traind=np.concatenate((data[t[0]], data[t[1]]), axis=0)
              labeltrain=np.concatenate((label[t[0]], label[t[1]]), axis=0)

              #testd=data[c]
              #labeltest=label[c]
              testd = d_test
              labeltest = l_test



              avg_cost=0
              auc_list=[]
              iterationSteps=0
              sess.run(iterator.initializer, feed_dict = {x: traind, y: labeltrain})
              try:

                  while iterationSteps <=100:
                          iterationSteps+=1
                      
                          ### Training
                          
                          _,lss=sess.run([optimizer,loss], feed_dict= {training: 1, dropprob: prob, beta1:beta_1, 
                                                                       beta2: beta_2, beta3:beta_3, learning_rate:lea_r, momentum_rate:mom_r,stdConv:stdc,stdNeu:stdn})
                        
                          
                          if iterationSteps % 10==0:
                                  ## Validation


                                  sess.run(iterator_val.initializer, feed_dict = {x: testd, y: labeltest})

                                  l,yl=sess.run([sigV, data_yV], feed_dict= {training: 0, dropprob: prob, beta1:beta_1, 
                                                                       beta2: beta_2, beta3:beta_3, learning_rate:lea_r, momentum_rate:mom_r,stdConv:stdc,stdNeu:stdn})
                                  auc=metrics.roc_auc_score(yl, l)
                                  print('AUC for the number of iterations',iterationSteps,'is:',auc)
                                  auc_list.append(auc)
                                  print(" running in %s second" % (time.time()-start_time))



              except tf.errors.OutOfRangeError:
                  pass
              print('===== Fold Done =====') 

              CV_auc_list.append(auc_list)

              
        print('The Cross Validation AUC for The Three Folds in 5 Different Iteration Steps:' , CV_auc_list)
        for i in range(len(auc_list)):
                Avg_List.append(np.mean([CV_auc_list[j][i] for j in range(len(CV_auc_list))]))
        print('The Average AUC for each Iteration Step of The Three Folds is:', Avg_List)
    
        
        maxlist=max(Avg_List)
        if maxlist>best_AUC:
          best_AUC=maxlist
          
          ind=Avg_List.index(maxlist)
          
          lr,mr,sc,sn,b1,b2,b3 = sess.run([learning_rate, momentum_rate,stdConv, stdNeu,beta1,beta2,beta3], feed_dict= {training: 0, dropprob: prob, beta1:beta_1, 
                                                                       beta2: beta_2, beta3:beta_3, learning_rate:lea_r, momentum_rate:mom_r,stdConv:stdc,stdNeu:stdn})
          print( 'Best hyperparameters So far:')
          print( 'Best Learning Rate', lr)
          print( 'Best Momentum Rate', mr)
          print( 'Best Learning Step', (ind+1)*4000)
          print( 'Best Sigma Conv', sc)
          print( 'Best Sigma NN', sn)
          print( 'Best Dropout Prob', prob)
          print( 'Best Beta 1', b1)
          print( 'Best Beta 2', b2)
          print( 'Best Beta 3', b3)
          
          save_LearningRate=lr
          save_Momentum=mr
          save_LearningStep=(ind+1)*4000
          save_SigmaConv=sc
          save_SigmaNeu=sn
          save_Dropprob=prob
          save_Beta1=b1
          save_Beta2=b2
          save_Beta3=b3




graph2=tf.Graph()
with graph2.as_default():
    
    num_input_channels=4
    num_filters=nummotif
    filter_shape=20
    pooling='max_pool'
    neuType='hidden'
    
    beta1=save_Beta1
    beta2=save_Beta2
    beta3=save_Beta3

    
    
    learning_rate= save_LearningRate
    momentum_rate= save_Momentum
    batch_size=1000
    
    
    with tf.device('/cpu:0'):
    
      x = tf.placeholder(tf.float32, [None, 147, 4],name='X')
      y = tf.placeholder(tf.float32,[None,1],name='y')


    with tf.device('/cpu:0'):
      
      #Set up iterator for the data
      dataset = tf.data.Dataset.from_tensor_slices((x, y))
      dataset = dataset.shuffle(500).repeat().batch(batch_size)
      iterator = dataset.make_initializable_iterator()
      data_X, data_y = iterator.get_next()
      data_y = tf.cast(data_y, tf.float32)

      dropprob = tf.placeholder_with_default(0.5, shape=(),name='prob')

      # Distinguish training and testing: training=1 for training , =0 for testing
      training = tf.placeholder_with_default(0.0, shape=(),name='training')
      
   
    with tf.device('/cpu:0'):
      
      conv_filt_shape = [filter_shape, num_input_channels, num_filters]

      stdConv=save_SigmaConv
      # initialise weights and bias for the filter
      conv_weights = tf.Variable(tf.truncated_normal(conv_filt_shape, mean=0,stddev=stdConv), name='Conv1_W')
      conv_bias = tf.Variable(tf.truncated_normal([num_filters]), name='Conv1_b')



      if pooling=='max_pool':
          W = tf.Variable(tf.truncated_normal([16,32], mean=0, stddev=0.3), name='W')
          b = tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.3), name='b')
      else:
          W = tf.Variable(tf.truncated_normal([32,32], mean=0, stddev=0.3), name='W')
          b = tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.3), name='b')     

      if neuType == 'nohidden':
          if pooling=='max_pool':
              wdim1=16
          else:
              wdim1=32
      else:
          wdim1=32
      stdNeu=save_SigmaNeu
      wd1 = tf.Variable(tf.truncated_normal([wdim1,1], mean=0, stddev=stdNeu), name='w2')
      bd1 = tf.Variable(tf.truncated_normal([1], mean=0, stddev=stdNeu), name='b2')


      xconv = convolution(data_X,num_input_channels,num_filters,filter_shape,conv_weights,conv_bias,wd1,bd1,W,b,pooling,neuType,training,dropprob)

      sig = tf.nn.sigmoid(xconv)
      if neuType == 'hidden':
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_y,logits=xconv))+ beta1*tf.norm(conv_weights,ord=1)+ beta2*tf.norm(wd1,ord=1)+ beta3*tf.norm(W,ord=1)
      else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_y,logits=xconv))+ beta1*tf.norm(conv_weights,ord=1)+ beta2*tf.norm(wd1,ord=1)

      optimizer=tf.train.MomentumOptimizer(learning_rate,momentum_rate,use_nesterov=True).minimize(loss)
    with tf.device('/cpu:0'):
      
      #Set up iterator for the validation data
      dataset_val = tf.data.Dataset.from_tensor_slices((x, y))
      dataset_val = dataset_val.batch(tf.cast(tf.size(y),tf.int64))
                  
      iterator_val = dataset_val.make_initializable_iterator()
      data_XV, data_yV = iterator_val.get_next()      
      data_yV = tf.cast(data_yV, tf.float32)
      
      
      data_XV = tf.placeholder_with_default(data_XV, shape=None, name='input')
      data_yV = tf.placeholder_with_default(data_yV, shape=None,name='label')
      
      
    with tf.device('/cpu:0'):
      xconvV = convolution(data_XV,num_input_channels,num_filters,filter_shape,conv_weights,conv_bias,wd1,bd1,W,b,pooling,neuType,training,dropprob)

      sigV = tf.nn.sigmoid(xconvV, name='Conv_V')
  
    saver = tf.train.Saver()

import copy
from sklearn import metrics
import numpy as np
import random



with tf.Session(graph=graph2, config=tf.ConfigProto(log_device_placement=True)) as sess:
    auc_list=[]
    best_auc=0
    for iter in range(6):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        

        prob=save_Dropprob
        iterationSteps=0
        sess.run(iterator.initializer, feed_dict = {x: data_all, y: label_all})
        try:

            while iterationSteps <=save_LearningStep:
                  iterationSteps+=1
              
                  ### Training
                  _,lss=sess.run([optimizer,loss], feed_dict= {training: 1, dropprob: prob})
                  
        except tf.errors.OutOfRangeError:
            pass
          
        ## Validation
        sess.run(iterator_val.initializer, feed_dict = {x: data_all, y: label_all})
        l,yl=sess.run([sigV,data_yV], feed_dict= {training: 0, dropprob: prob})

        auc=metrics.roc_auc_score(yl, l)
        print('AUC of Model Num',iter,' is : ', auc)
        
        if auc > best_auc:
          best_auc=auc
          print('Best AUC So Far is : ', best_auc)
          ##save model
          #save_path = saver.save(sess, "/content/drive/My Drive/Colab Notebooks/Test2/model2")
          save_path = saver.save(sess, "./model2")
          
          print('Model Saved!')

import copy
from sklearn import metrics
import numpy as np
import random

#filename='/content/drive/My Drive/Colab Notebooks/Chip-seq/ELK1_GM12878_ELK1_(1277-1)_Stanford_B.seq.gz'
#filename='./data/encode/ARID3A_K562_ARID3A_(sc-8821)_Stanford_AC.seq'
filename = "./data/encode/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_B.seq"
test= ChipTest(filename)
dataAll =test.openFile()
data_all=np.asarray([el[0] for el in dataAll],dtype=np.float32)
label_all=np.asarray([el[1] for el in dataAll],dtype=np.float32).reshape(len(data_all),1)

import tensorflow as tf

TestGraph=tf.Graph()
with tf.Session(graph = TestGraph) as sess:    
  
  # #First let's load meta graph and restore weights
  ckpt = tf.train.get_checkpoint_state('./Test2', latest_filename='checkpoint')
  
  if ckpt and ckpt.model_checkpoint_path:  # if there's checkpoint
    saver = tf.train.import_meta_graph('./data/encode/Test2/model2.meta')
    saver.restore(sess, ckpt.model_checkpoint_path)


    X = TestGraph.get_tensor_by_name("input:0")
    y = TestGraph.get_tensor_by_name("label:0")


    training = TestGraph.get_tensor_by_name("training:0")
    prob = TestGraph.get_tensor_by_name("prob:0")

    # #Now, access the op to run. 
    Conv_V = TestGraph.get_tensor_by_name("Conv_V:0")
    feed_dict2={X:data_all,y:label_all,prob:save_Dropprob}
    l=sess.run(Conv_V,feed_dict2)
    auc=metrics.roc_auc_score(label_all, l)
    print(auc)




