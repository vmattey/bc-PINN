# -*- coding: utf-8 -*-
"""
bc-PINN for Allen Cahn 1D 

1. With ICGL (10% ADAM Iterations)
2. Initialization of parameters obtained from previous segment
3. Weighting of loss function with maximum order of derivatives and delta_x

"""

# %tensorflow_version 1.15
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import os
import pickle
import errno
import math

np.random.seed(1234)
tf.set_random_seed(1234)

# Initialize the class
class PhysicsInformedNN:
    def __init__(self, x0, u0, tb, X_f, X_star, u_star, layers, lb, ub):
        
        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        self.lb = lb
        self.ub = ub
        self.nb = np.size(tb)
               
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]
        self.u0 = u0

        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.x_star = X_star[:,0:1]
        self.t_star = X_star[:,2:3]
        self.u_star = u_star
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Initialize NNs
        self.wi = tf.constant(1,dtype=tf.float32)
        self.wb = tf.constant(1,dtype=tf.float32)
        self.wr = tf.constant(1,dtype=tf.float32)
        self.delta = tf.constant(1.35, dtype = tf.float32)
        
        # tf Placeholders        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        
        self.x_star_tf = tf.placeholder(tf.float32, shape=[None, self.x_star.shape[1]]) # MODIFIED FOR I.C at all time steps
        self.t_star_tf = tf.placeholder(tf.float32, shape=[None, self.t_star.shape[1]]) # MODIFIED FOR I.C at all time steps
        self.u_star_tf = tf.placeholder(tf.float32, shape=[None, self.u_star.shape[1]]) # MODIFIED FOR I.C at all time steps

        
        # tf Graphs
        self.u0_pred, _ = self.net_uv(self.x0_tf, self.t0_tf)
        self.u_lb_pred, self.u_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.u_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)
        
        self.u_star_pred, _ = self.net_uv(self.x_star_tf, self.t_star_tf)
        
        
        # Loss
        
        self.loss = 64*tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred))
                   

        self.loss_star = tf.reduce_mean(tf.square(self.u_star_tf - self.u_star_pred))
                    
        self.loss_ui = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred) + tf.abs(self.u0_tf - self.u0_pred))
                       
        self.loss_ub = tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred) + tf.abs(self.u_lb_pred - self.u_ub_pred))+ \
                       tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred) + tf.abs(self.u_lb_pred - self.u_ub_pred))
                       
        self.loss_ur = tf.reduce_mean(tf.square(self.f_u_pred) + tf.abs(self.f_u_pred))

        

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 10000,
                                                                           'maxfun': 20000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.initial_learning_rate_star = 0.01
        self.initial_learning_rate = 0.01
        self.global_step_star = tf.Variable(0,trainable=False)
        self.global_step = tf.Variable(0,trainable=False)
        
        increment_global_step_star = tf.assign(self.global_step_star, self.global_step_star + 1)
        increment_global_step = tf.assign(self.global_step, self.global_step + 1)
        
        decayed_lr_star = tf.train.exponential_decay(self.initial_learning_rate_star,
                                                self.global_step_star, 2000,
                                                0.1, staircase=True)

        decayed_lr = tf.train.exponential_decay(self.initial_learning_rate,
                                                self.global_step, 2000,
                                                0.2, staircase=True)
        
        self.optimizer_Adam_star = tf.train.AdamOptimizer(learning_rate=decayed_lr_star)
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam_star = self.optimizer_Adam_star.minimize(self.loss_star,global_step=self.global_step_star)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uv(self, x, t):
        X = tf.concat([x,t],1)
        
        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:,0:1]
        u_x = tf.gradients(u, x)[0]
        return u, u_x

    def net_f_uv(self, x, t):
        u, u_x = self.net_uv(x,t)
        
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x,x)[0]
        
        fu = u_t - 0.0001*u_xx + 5*u*(u-1)*(u+1)
        
        return fu
    
    
    def callback(self, loss, loss_ui, loss_ub, loss_ur):
        self.loss_lbfgs.append(loss)
        self.loss_ui_lbfgs.append(loss_ui)
        self.loss_ub_lbfgs.append(loss_ub)
        self.loss_ur_lbfgs.append(loss_ur)

        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.x_lb_tf: self.x_lb, 
                   self.t_lb_tf: self.t_lb, self.x_ub_tf: self.x_ub, 
                   self.t_ub_tf: self.t_ub, self.x_f_tf: self.x_f, 
                   self.t_f_tf: self.t_f}
        
        tf_dict_star = {self.x_star_tf: self.x_star, self.t_star_tf: self.t_star, 
                        self.u_star_tf: self.u_star}
       
        self.loss_adam = []
        self.loss_ui_adam = []
        self.loss_ur_adam = []
        self.loss_ub_adam = []


        start_time = time.time()
        for it in range(nIter):
          if it <= nIter/10 :
            self.sess.run(self.train_op_Adam_star, tf_dict_star)
            if it % 10 == 0:
              elapsed = time.time() - start_time
              loss_value = self.sess.run(self.loss_star, tf_dict_star)
              step_star = self.sess.run(self.global_step_star)
              step = self.sess.run(self.global_step)
             
              #learning_rate_value = self.sess.run(self.optimizer_Adam._lr)
              self.loss_adam.append(loss_value)
              print('It: %d, Loss: %.3e, Time: %.2f, step_star: %d' % 
                    (it, loss_value, elapsed, step_star))
              
              
              start_time = time.time()
          else:
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
              elapsed = time.time() - start_time
              loss_value = self.sess.run(self.loss, tf_dict)
              loss_value_ui = self.sess.run(self.loss_ui, tf_dict)
              loss_value_ub = self.sess.run(self.loss_ub, tf_dict)
              loss_value_ur = self.sess.run(self.loss_ur, tf_dict)

              self.loss_adam.append(loss_value)
              self.loss_ui_adam.append(loss_value_ui)
              self.loss_ub_adam.append(loss_value_ub)
              self.loss_ur_adam.append(loss_value_ur)

              print('It: %d, Loss: %.3e, Loss_ui: %.3e, Loss_ub: %.3e, Loss_ur: %.3e, Time: %.2f, step_star: %d, step: %d' % 
                 (it, loss_value, loss_value_ui, loss_value_ub, loss_value_ur, elapsed, step_star, step))
              start_time = time.time()
                                          
        self.loss_lbfgs = []
        self.loss_ui_lbfgs = []
        self.loss_ub_lbfgs = []
        self.loss_ur_lbfgs = []
                                                                             
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss,self.loss_ui,self.loss_ub,self.loss_ur], 
                                loss_callback = self.callback)
                                    
    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u0_pred, tf_dict)   
             
        return u_star

np.random.seed(1234)
tf.set_random_seed(1234)

# Initialize the class
class bcPhysicsInformedNN:
    def __init__(self, x0, u0, t0, u1, tb, X_old, X_f, X_star, u_star, layers, lb, ub, weights, biases):
        
        X0 = np.concatenate((x0, t0*np.ones(x0.shape)), 1) # (x0, t0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        
        self.lb = lb
        self.ub = ub
        self.nb = np.size(tb)
               
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]
        self.u0 = u0

        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.x_star = X_star[:,0:1]
        self.t_star = X_star[:,2:3]

        self.u_star = u_star

        self.x1 = X_old[:,0:1]
        self.t1 = X_old[:,1:2]
        
        self.u1 = u1

        
        # Initialize NNs
        self.layers = layers
        #self.weights, self.biases = self.initialize_NN(layers)
        self.weights = weights
        self.biases = biases
        
        # Initialize NNs
        self.wi = tf.constant(1,dtype=tf.float32)
        self.wb = tf.constant(1,dtype=tf.float32)
        self.wr = tf.constant(1,dtype=tf.float32)
        self.ws = tf.constant(1,dtype=tf.float32)
        self.delta = tf.constant(1.35,dtype=tf.float32)
        
        # tf Placeholders        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        
        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x1.shape[1]])
        self.t1_tf = tf.placeholder(tf.float32, shape=[None, self.t1.shape[1]])

        self.u1_tf = tf.placeholder(tf.float32, shape=[None, self.u1.shape[1]])
        
        self.x_star_tf = tf.placeholder(tf.float32, shape=[None, self.x_star.shape[1]]) # MODIFIED FOR I.C at all time steps
        self.t_star_tf = tf.placeholder(tf.float32, shape=[None, self.t_star.shape[1]]) # MODIFIED FOR I.C at all time steps
        self.u_star_tf = tf.placeholder(tf.float32, shape=[None, self.u_star.shape[1]]) # MODIFIED FOR I.C at all time steps

        
        # tf Graphs
        self.u0_pred, _ = self.net_uv(self.x0_tf, self.t0_tf)
        self.u1_pred, _ = self.net_uv(self.x1_tf,self.t1_tf)
        self.u_lb_pred, self.u_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.u_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)
        
        self.u_star_pred, _ = self.net_uv(self.x_star_tf, self.t_star_tf)
        
        
        # Loss


        self.loss = 64*tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    64*tf.reduce_mean(tf.square(self.u1_pred - self.u1_tf))
        


        self.loss_star = tf.reduce_mean(tf.square(self.u_star_tf - self.u_star_pred))
                    
        self.loss_ui = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred) + tf.abs(self.u0_tf - self.u0_pred))
                       
        self.loss_ub = tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred) + tf.abs(self.u_lb_pred - self.u_ub_pred))+ \
                       tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred) + tf.abs(self.u_lb_pred - self.u_ub_pred))
                       
        self.loss_ur = tf.reduce_mean(tf.square(self.f_u_pred) + tf.abs(self.f_u_pred))
        self.loss_us = tf.reduce_mean(tf.square(self.u1_pred - self.u1_tf) + tf.abs(self.u1_pred - self.u1_tf))

        self.loss_hui = tf.abs(self.u0_tf - self.u0_pred)
                       
        self.loss_hub = tf.abs(self.u_lb_pred - self.u_ub_pred)+ \
                       tf.abs(self.u_x_lb_pred - self.u_x_ub_pred)
                       
        self.loss_hur = tf.abs(self.f_u_pred)

        self.loss_hus = tf.abs(self.u1_pred - self.u1_tf)


        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 10000,
                                                                           'maxfun': 20000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.initial_learning_rate_star = 0.01
        self.initial_learning_rate = 0.01
        self.global_step_star = tf.Variable(0,trainable=False)
        self.global_step = tf.Variable(0,trainable=False)
        
        increment_global_step_star = tf.assign(self.global_step_star, self.global_step_star + 1)
        increment_global_step = tf.assign(self.global_step, self.global_step + 1)
        
        decayed_lr_star = tf.train.exponential_decay(self.initial_learning_rate_star,
                                                self.global_step_star, 2000,
                                                0.1, staircase=True)

        decayed_lr = tf.train.exponential_decay(self.initial_learning_rate,
                                                self.global_step, 2000,
                                                0.2, staircase=True)
        
        self.optimizer_Adam_star = tf.train.AdamOptimizer(learning_rate=decayed_lr_star)
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam_star = self.optimizer_Adam_star.minimize(self.loss_star,global_step=self.global_step_star)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uv(self, x, t):
        X = tf.concat([x,t],1)
        
        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:,0:1]
        u_x = tf.gradients(u, x)[0]
        return u, u_x

    def net_f_uv(self, x, t):
        u, u_x = self.net_uv(x,t)
        
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x,x)[0]
        
        fu = u_t - 0.0001*u_xx + 5*u*(u-1)*(u+1)
        return fu
    
    def callback(self, loss, loss_ui, loss_ub, loss_ur, loss_us):
        self.loss_lbfgs.append(loss)
        self.loss_ui_lbfgs.append(loss_ui)
        self.loss_ub_lbfgs.append(loss_ub)
        self.loss_ur_lbfgs.append(loss_ur)
        self.loss_us_lbfgs.append(loss_us)

        print('Loss:', loss)

    def train(self, nIter):
        
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.x_lb_tf: self.x_lb, 
                   self.t_lb_tf: self.t_lb, self.x_ub_tf: self.x_ub, 
                   self.t_ub_tf: self.t_ub, self.x_f_tf: self.x_f, 
                   self.t_f_tf: self.t_f, self.x1_tf: self.x1, 
                   self.t1_tf: self.t1, self.u1_tf: self.u1}
       
        tf_dict_star = {self.x_star_tf: self.x_star, self.t_star_tf: self.t_star, 
                        self.u_star_tf: self.u_star}
       
        self.loss_adam = []
        self.loss_ui_adam = []
        self.loss_ub_adam = []
        self.loss_ur_adam = []
        self.loss_us_adam = []

        start_time = time.time()
        for it in range(nIter):
          if it <= nIter/10 :
            self.sess.run(self.train_op_Adam_star, tf_dict_star)
            if it % 10 == 0:
              elapsed = time.time() - start_time
              loss_value = self.sess.run(self.loss_star, tf_dict_star)
              self.loss_adam.append(loss_value)
              print('It: %d, Loss: %.3e, Time: %.2f' % 
                    (it, loss_value, elapsed))
              start_time = time.time()
          else:
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
              elapsed = time.time() - start_time
              loss_value = self.sess.run(self.loss, tf_dict)
              loss_value_ui = self.sess.run(self.loss_ui, tf_dict)
              loss_value_ub = self.sess.run(self.loss_ub, tf_dict)
              loss_value_ur = self.sess.run(self.loss_ur, tf_dict)
              loss_value_us = self.sess.run(self.loss_us, tf_dict)

              self.loss_adam.append(loss_value)
              self.loss_ui_adam.append(loss_value_ui)
              self.loss_ub_adam.append(loss_value_ub)
              self.loss_ur_adam.append(loss_value_ur)
              self.loss_us_adam.append(loss_value_us)
              
              print('It: %d, Loss: %.3e, Loss_ui: %.3e, Loss_ub: %.3e, Loss_ur: %.3e, Loss_us: %.3e, Time: %.2f' % 
                 (it, loss_value, loss_value_ui, loss_value_ub, loss_value_ur, loss_value_us, elapsed))
              start_time = time.time()
                                          
        self.loss_lbfgs = []
        self.loss_ui_lbfgs = []
        self.loss_ub_lbfgs = []
        self.loss_ur_lbfgs = []
        self.loss_us_lbfgs = []  
                                                                       
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss, self.loss_ui, self.loss_ub, self.loss_ur, self.loss_us], 
                                loss_callback = self.callback)
                                    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u0_pred, tf_dict)   

        return u_star

if __name__ == "__main__": 
     
    noise = 0.0        

    #layers = [2, 200, 200, 200, 200, 1]
    #layers = [2, 256, 256, 256, 256, 1]
    layers = [2, 128, 128, 128, 128, 128, 128, 1]
    path = './AC_C1_0001_ICGL_TL'
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory not created')
        else:
            raise

        
    data = scipy.io.loadmat('./AC_R_1.mat')
    
    t = data['tt'].flatten()[:,None]
    #x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Nx = 128
    x = np.linspace(-1,1,Nx)
    x = x.flatten()[:,None]
    dx = x[2] - x[1]
    
    x0 = x
    u0 = (x**2)*np.cos(np.pi*x)
    #u0 = Exact_u[:,0]
    u0 = u0.reshape(x.shape)
    

    start  = 0
    step = 50
    stop = 201
    steps_lb = np.arange(0,stop+step,step)
    steps_ub = 1 + steps_lb
    
    iterations = 10000
    N_f = 20000 
    counter = 0

    for i in range(0,steps_lb.size-1):
        t1 = steps_lb[i]
        t2 = steps_ub[i+1]
        temp_t = t[:t2,:]
        t = t[t1:t2,:]
        
        ############### For implementing ICGL ##############
        X, T = np.meshgrid(x,t)
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        
        ############### Modified only 50% of the total segment size##############
        N_b = int(step/2)
        idx_t = np.random.choice(t.shape[0], N_b, replace=False)
        tb = t[idx_t,:]
                    
        
        if counter == 0:
            # Domain bounds
            lb = np.array([-1, 0.0])
            ub = np.array([1, np.max(t)])
            
            temp_lb = np.array([-1, np.min(t)])
            temp_ub = np.array([1, np.max(t)])
            
            X_f = temp_lb +  (temp_ub-temp_lb)*lhs(2,N_f)
            u_star = np.tile(u0,(len(t),1))
            
            globals()[f"model_{counter}"] = PhysicsInformedNN(x0, u0, tb, X_f, X_star, u_star, layers, lb, ub)
            start_time = time.time()                
            globals()[f"model_{counter}"].train(iterations)
            elapsed = time.time() - start_time                
            print('Training time: %.4f' % (elapsed))
            x_old, t_old = np.meshgrid(x,temp_t)
            X_old = np.hstack((x_old.flatten()[:,None], t_old.flatten()[:,None]))
            pred = globals()[f"model_{counter}"].predict(X_old)
            u1 = pred
        else:
            # Domain bounds
            # For any step other than initial step
            lb = np.array([-1, 0.0])
            ub = np.array([1, np.max(t)])
            
            temp_lb = np.array([-1, np.min(t)])
            temp_ub = np.array([1, np.max(t)])
            
            u0 = u1[-Nx:]
            t0 = t[0]
            
            X_f = temp_lb + (temp_ub - temp_lb)*lhs(2,N_f)
            u_star = np.tile(u0,(len(t),1))
            
            globals()[f"model_{counter}"] = bcPhysicsInformedNN(x0, u0, t0, u1, tb, X_old, X_f, X_star, u_star, layers, lb, ub, W, b)
            start_time = time.time()                
            globals()[f"model_{counter}"].train(iterations)
            elapsed = time.time() - start_time                
            print('Training time: %.4f' % (elapsed))
            x_old, t_old = np.meshgrid(x,temp_t)
            X_old = np.hstack((x_old.flatten()[:,None], t_old.flatten()[:,None]))
            pred = globals()[f"model_{counter}"].predict(X_old)
            u1 = pred
            
                                   
        layers_freeze = 2
        weights = globals()[f"model_{counter}"].sess.run(globals()[f"model_{counter}"].weights)
        biases = globals()[f"model_{counter}"].sess.run(globals()[f"model_{counter}"].biases)
        W = [] 
        b = []
        for i in range(len(weights)):
            if i < layers_freeze:
                W.append(tf.Variable(weights[i],trainable='False'))
                b.append(tf.Variable(biases[i],trainable='False'))
            else:
                W.append(tf.Variable(weights[i],trainable='True'))
                b.append(tf.Variable(biases[i],trainable='True'))
        
        
        t = data['tt'].flatten()[:,None]
        
        
        os.chdir(path)
        name = 'Sequential_training_' + str(t1) + '_' + str(t2)
        path1 = path + '/' + name
        os.makedirs(name)
        os.chdir(path1)
        #saver = tf.train.Saver()
        #saver.save( globals()[f"model_{counter}"].sess,'model')

        file_name = "biases.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(biases, open_file)
        open_file.close()

        file_name = "weights.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(weights, open_file)
        open_file.close()

        file_name = "u1.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(u1, open_file)
        open_file.close()
        
        file_name = "MSE_ADAM.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(globals()[f"model_{counter}"].loss_adam, open_file)
        open_file.close()

        file_name = "MSE_LBFGS.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(globals()[f"model_{counter}"].loss_lbfgs, open_file)
        open_file.close()

        file_name = "MSE_UI_ADAM.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(globals()[f"model_{counter}"].loss_ui_adam, open_file)
        open_file.close()

        file_name = "MSE_UB_ADAM.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(globals()[f"model_{counter}"].loss_ub_adam, open_file)
        open_file.close()

        file_name = "MSE_UR_ADAM.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(globals()[f"model_{counter}"].loss_ur_adam, open_file)
        open_file.close()

        file_name = "MSE_UI_LBFGS.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(globals()[f"model_{counter}"].loss_ui_lbfgs, open_file)
        open_file.close()

        file_name = "MSE_UB_LBFGS.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(globals()[f"model_{counter}"].loss_ub_lbfgs, open_file)
        open_file.close()

        file_name = "MSE_UR_LBFGS.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(globals()[f"model_{counter}"].loss_ur_lbfgs, open_file)
        open_file.close()


        if counter >= 1:

          file_name = "MSE_US_ADAM.pkl"
          open_file = open(file_name, "wb")
          pickle.dump(globals()[f"model_{counter}"].loss_us_adam, open_file)
          open_file.close()
          
          file_name = "MSE_US_LBFGS.pkl"
          open_file = open(file_name, "wb")
          pickle.dump(globals()[f"model_{counter}"].loss_us_lbfgs, open_file)
          open_file.close()


        os.chdir(path)

        counter += 1 

