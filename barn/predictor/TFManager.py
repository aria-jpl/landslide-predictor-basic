import numpy as np
import sys
import os
import pickle as cp
import tensorflow as tf
from binary_data_handler import BinaryDataHandler
import argparse
import json
#from matplotlib import pyplot as plt
#import matplotlib as mpl
#mpl.rcParams['image.cmap'] = 'jet'
import importlib
import shutil
from datetime import datetime
import types
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from sympy.ntheory import factorint

class TFManager():
    def __init__(self):
        self._summ_list = ['mean','stddev','hist']
        self._data_handler = None
        self._inference_ops = []
        self._inference_ops_dict = {}
        self._inference_ops_cnt_dict = {}
        #keep track of layers that have been given names to apply shortcut
        self._named_layers = {}
        self._weights_list = []
        self._weights_cnt = 0
        self._biases_cnt = 0
        self._relu_cnt = 0
        self._latest_checkpoint = ''
        self._class_thr = -1#set to some value if want to use a certain threshold
                              #for classification instead of argmax

    def predict(self,inps):
        with tf.Graph().as_default():
            typel = inps['loss_type']
            model_id = inps['model_id']
            log_dir = inps['log_dir'] + '_' + str(model_id)
            batch_size = inps['batch_size']
            pred_dir = inps['pred_dir']
            os.environ["CUDA_VISIBLE_DEVICES"]= inps['visible_gpu']           
            if 'init_file' in inps:
                init_file = inps['init_file']
            else:
                init_file = ''
            class_thr = inps['class_thr']
            self._class_thr = class_thr
            
            imports = None if "imports" not in inps else inps["imports"]
            self.set_data_handler(imports)
            test_handler = self._data_handler(batch_size,shuffle=False,init_file=init_file)
            self.load_data(inps,test_handler,2)        
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = inps['gpu_frac']
            data_type = test_handler._data.dtype.name
            inp_shapes = test_handler._data.shape[1:]
            shape = [None]
            shape.extend(inp_shapes)
            tr_x = tf.placeholder(data_type, shape, name='x-input')
            keep_prob = tf.placeholder(tf.float32)
            is_test = tf.placeholder(tf.bool)
            
            if isinstance(inps['layers'],list):
                y = self.inference(inps, tr_x, is_test, keep_prob)
            elif isinstance(inps['layers'],dict):
                self.inference = types.MethodType( getattr(importlib.import_module(inps['layers']['from']),inps['layers']['import']), self )        
                y = self.inference(inps, tr_x, is_test, keep_prob)
            else:
                print('The layer input must be a list of dict')
            saver = tf.train.Saver()
            lowest_error = sys.float_info.max
            
            if inps['gpu_frac'] < 0:
                config = tf.ConfigProto(device_count = {'GPU': 0})
            elif inps['gpu_frac'] < .9999:
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = inps['gpu_frac']
            else:
                config = None
            
            with tf.Session(config=config) as sess:
                latest = tf.train.latest_checkpoint(log_dir,latest_filename='checkpoint')
                if isinstance(latest,list):
                    latest = latest[0]
                if latest != self._latest_checkpoint:
                    global_step = int(latest.split('/')[-1].split('-')[-1])
                    saver.restore(sess, latest)
                    ret = [] 
                    fname = os.path.join(pred_dir,inps['data_file'][2].split('/')[-1])
                    sp = fname.split('.')
                    sp[-1] = 'pred'
                    fname = '.'.join(sp)
                    nbatches = int(test_handler._data_size/batch_size) + 1 if (test_handler._data_size%batch_size) > 0 else 0
                    start_t = datetime.now()
                    for i in range(nbatches):                     
                        feed_dict = self.inference_fill_feed_dict(test_handler,tr_x,keep_prob,is_test)
                        yv = sess.run(y,feed_dict=feed_dict)
                        ypred = i*batch_size + self.get_predictions(yv, class_thr)
                        if i == nbatches - 1 and (test_handler._data_size%batch_size) > 0:
                            pred = self.get_predictions(yv, class_thr)
                            ypred = i*batch_size + pred[pred < (test_handler._data_size%batch_size)]
                        ret.extend(ypred)
                    ret = np.array(ret,np.int32)
                    ret.tofile(fname)
                    print("fname", fname)
                    print("ret.shape", ret.shape)
                    print("ret.dtype", ret.dtype)
                    print("ret", ret)
                    print(datetime.now() - start_t)
                    
    def get_predictions(self,y,class_thr=-1):
        if class_thr > 0:
            assert class_thr < 1,'Threshold must be between 0 and 1'
            y = np.exp(y)/np.tile(np.sum(np.exp(y),1).reshape([-1,1]),[1,2])
            ym = (y[:,1] >= class_thr).astype(np.int)
        else:
            ym = np.argmax(y,1)
        sel = np.nonzero(ym == 1)[0].astype(np.int32)
        return sel
        
    
    def get_batch_size(self,data_size,batch_size):
        fact = factorint(data_size)
        keys = np.sort(np.array(list(fact.keys())))
        if len(keys) > 1:
            nbatch_size = 1
            found = False
            for k in keys:
                for _ in range(fact[k]):
                    nbatch_size *= k
                    if nbatch_size > 2*batch_size:
                        nbatch_size /= k
                        found = True
                        break
                if found:
                    break
        else:
            nbatch_size = batch_size
        return int(nbatch_size)


    def inference_fill_feed_dict(self,datain,data_ph,keep_prop_ph,is_test_ph):                  
        data_,dum,dum = datain.next_batch()
        feed_dict = {
                     data_ph:data_,
                     keep_prop_ph:1,
                     is_test_ph:True
                     }
        return feed_dict

    def run(self,inps):
        with tf.Graph().as_default():
            typel = inps['loss_type']
            model_id = inps['model_id']
            log_dir = inps['log_dir'] + '_' + str(model_id)
            split = inps['split']
            batch_size = inps['batch_size']
            valid_batch_size = inps['validation_batch_size']
            print_summary = inps['summary']
            save_best = inps['save_best']
            save_met = inps['save_metric']
            if save_met == 'accuracy':
                save_met_indx = 2 
            elif save_met == 'precision':
                save_met_indx = 0
            elif save_met == 'recall':
                save_met_indx = 1
            elif save_met == 'f1':
                save_met_indx = 3
            else:
                print("Unrectognized option",save_met)
                raise Exception
            print_summary_test = inps['summary_test']
            max_steps = inps['max_steps']
            keep_prob_val = inps['keep_prob']
            summary_freq = inps['summary_freq']
            checkpoint_freq = inps['checkpoint_freq']
            is_train = inps['is_train']
            reload = inps['reload']    
            os.environ["CUDA_VISIBLE_DEVICES"]= inps['visible_gpu']
            if inps['rm_log_dir'] and is_train and not reload:
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir)
            opt = inps['optimizer']
            oargs = tuple(inps['optimizer_args'])
            if inps['std_to_file']:
                sys.stdout = open(inps['std_file'],'w')
            if 'optimizer_kwargs' in inps:
                okwargs = inps['optimizer_kwargs']
            else:
                okwargs = {}
            debug = inps['debug']
            Nav = inps['nema']
            alpha = 2/(Nav + 1)
            if 'init_file' in inps:
                init_file = inps['init_file']
            else:
                init_file = ''
            class_thr = inps['class_thr']
            self._class_thr = class_thr
            
            imports = None if "imports" not in inps else inps["imports"]
            self.set_data_handler(imports)
            test_handler = self._data_handler(batch_size,shuffle=True,init_file=init_file)
            self.load_data(inps,test_handler,is_train)        
            train = self._data_handler(batch_size,shuffle=True,init_file=init_file)
            valid = self._data_handler(valid_batch_size,shuffle=True,init_file=init_file)
            valid._validation_batch_size = valid_batch_size
            self.load_data(inps,train,is_train)        
            train.split_data(valid,split,shuffle=True)
          
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = inps['gpu_frac']
            data_type = train._data.dtype.name
            labels_type = train._labels.dtype.name
            if train._weights is not None:
                shape = [train._data.shape[1:],train._labels.shape[1:],train._weights.shape[1:]]
            else:
                shape = [train._data.shape[1:],train._labels.shape[1:]]
            tr_x,tr_y,tr_w,keep_prob,is_test = self.get_place_holders(shape,data_type,labels_type)
            
            if isinstance(inps['layers'],list):
                y = self.inference(inps, tr_x, is_test, keep_prob)
            elif isinstance(inps['layers'],dict):
                self.inference = types.MethodType( getattr(importlib.import_module(inps['layers']['from']),inps['layers']['import']), self )        
                y = self.inference(inps, tr_x, is_test, keep_prob)
            else:
                print('The layer input must be a list of dict')

            loss = self.loss(typel,y,tr_y,tr_w)
            #get variables for batch normalization and update
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #with tf.control_dependencies(update_ops):
            #    train_step = self.training(opt,loss,*oargs,**okwargs)
            train_step = self.training(opt,loss,*oargs,**okwargs)
            merged = tf.summary.merge_all()
            saver = tf.train.Saver()
            lowest_error = sys.float_info.max
            if is_train:
                #if fraction is one, let tf decide since 100% might not be available
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                
                sess = tf.InteractiveSession(config=config)
                if reload:
                    latest = tf.train.latest_checkpoint(log_dir,latest_filename='checkpoint')
                    if isinstance(latest,list):
                        latest = latest[0]
                    if latest != self._latest_checkpoint:
                        global_step = int(latest.split('/')[-1].split('-')[-1])
                        saver.restore(sess, latest)
                train_writer = tf.summary.FileWriter(os.path.join(log_dir,'train'), sess.graph)
                if not reload:
                    tf.global_variables_initializer().run() 
                if debug:
                    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)          
                best = 0
                if save_best == 1:
                    shutil.rmtree(os.path.join(log_dir,'saved_model'),True)
                    os.mkdir(os.path.join(log_dir,'saved_model'))
                    fp = open(os.path.join(log_dir,'saved_model','performance.log'),'w')
                for i in range(max_steps):
                    feed_dict = self.fill_feed_dict(train,keep_prob_val,False,tr_x,tr_y,tr_w,keep_prob,is_test)
                    _,lossv = sess.run([train_step,loss],feed_dict=feed_dict)

                    #summary,_,lossv = sess.run([merged,train_step,loss],feed_dict={keep_prob:0.5,is_test:False})
                    #train_writer.add_summary(summary, i)
                    try:
                        lossavg = alpha*lossv + (1-alpha)*lossavg
                    except:
                        lossavg = lossv
                    if (i+1)%summary_freq == 0:
                        if print_summary_test:
                            loop_vals = [[train,keep_prob_val,False],[valid,1,True]]
                        else:
                            loop_vals = [[train,keep_prob_val,False]]
                        tog = 0
                        stage = ['train','validation']
                        for data_t, keep_prob_now,is_test_now in loop_vals:
                            feed_dict = self.fill_feed_dict(data_t,keep_prob_now,is_test_now,tr_x,tr_y,tr_w,keep_prob,is_test)
        
                            if print_summary  and typel == 'sparse_cross_entropy':
                                lossv,yv,yt = sess.run([loss,y,tr_y],feed_dict=feed_dict)
                                ret = self.get_error(typel,yv,yt)
                                if(len(ret) == 4):#is precision and recall
                                    precision = ret[0]
                                    recall = ret[1]
                                    accuracy = ret[2]
                                    f1s = ret[3]
                                    if save_best == 1 and stage[tog%2] == 'validation':
                                        met = ret[save_met_indx]
                                        if met >= best:#use equal since later ones have seen more data
                                            best = met
                                            yv.tofile(os.path.join(log_dir,'best_model_'  + save_met+ '_%.2f.dat')%(met))
                                            checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                                            saver.save(sess, checkpoint_file, global_step=i,latest_filename='checkpoint')
                                        fp.write(('i = %d , loss  = %.2f, accuracy = %.2f, precision = %.2f, recall = %.2f, F1 score = %.2f time = %s\n')%(i,lossv,accuracy,precision,recall,f1s,datetime.now()))
                                    print(('i = %d , %s loss  = %.2f, accuracy = %.2f, precision = %.2f, recall = %.2f, F1 score = %.2f time = %s')%(i,stage[tog%2],lossv,accuracy,precision,recall,f1s,datetime.now()))
                                else:
                                    print(('i = %d, %s loss = %.2f, error = %.2f, time = %s')%(i,stage[tog%2], lossv, ret, datetime.now()))
                            else:
                                lossv,xv,yv,yt = sess.run([loss,tr_x,y,tr_y],feed_dict=feed_dict)
                                print(('i = %d, %s loss = %.2f, time = %s')%(i,stage[tog%2], lossv**.5, datetime.now())) 
                            tog += 1
                    if ((((i + 1) % checkpoint_freq) == 0 or (i + 1) == max_steps) and save_best == 0):
                        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=i,latest_filename='checkpoint')
                if save_best == 1:
                    fp.close()
                train_writer.close()
            else:
                if inps['gpu_frac'] < 0:
                    config = tf.ConfigProto(device_count = {'GPU': 0})
                elif inps['gpu_frac'] < .9999:
                    config = tf.ConfigProto()
                    config.gpu_options.per_process_gpu_memory_fraction = inps['gpu_frac']
                else:
                    config = None
                with tf.Session(config=config) as sess:
                    latest = tf.train.latest_checkpoint(log_dir,latest_filename='checkpoint')
                    if isinstance(latest,list):
                        latest = latest[0]
                    if latest != self._latest_checkpoint:
                        global_step = int(latest.split('/')[-1].split('-')[-1])
                        saver.restore(sess, latest)
                        avgs = np.zeros(5)
                        if test_handler._batch_size == -1:
                            tot_iter = 1
                        else:
                            tot_iter = int(test_handler._labels.shape[0]/batch_size)  
                        stats = np.zeros([tot_iter,4])
                        for i in range(tot_iter):                    
                            feed_dict = self.fill_feed_dict(test_handler,1,True,tr_x,tr_y,tr_w,keep_prob,is_test)
                            if typel == 'sparse_cross_entropy':
                                lossv,yv,yt = sess.run([loss,y,tr_y],feed_dict=feed_dict)
                                ret = self.get_error(typel,yv,yt,'rates')
                                if(len(ret) == 4):#is precision and recall
                                    precision = ret[0]
                                    recall = ret[1]
                                    accuracy = ret[2]
                                    f1s = ret[3]
                                    stats[i] = np.array(ret)
                                    avgs[4] += lossv
                                    #print(('i = %d , loss  = %.2f, accuracy = %.2f, precision = %.2f, recall = %.2f, F1 score = %.2f time = %s')%(i*batch_size,lossv,accuracy,precision,recall,f1s,datetime.now()))
                                else:
                                    print(('i = %d, loss = %.2f, error = %.2f, time = %s')%(global_step, lossv, ret, datetime.now()))
                            else:
                                lossv,xv,yv,yt = sess.run([loss,tr_x,y,tr_y],feed_dict=feed_dict)
                                avgs[0] += lossv
                                #print(('i = %d, loss valid = %.2f, time = %s')%(global_step, lossv, datetime.now())) 
                        if typel == 'sparse_cross_entropy':
                            avgs[4] /= tot_iter
                            tp,fp,tn,fn = stats.sum(0)
                            try:
                                prec = tp/(tp + fp)
                            except:
                                prec = 0
                            try:
                                rec = tp/(tp + fn)
                            except:
                                rec = 0
                            try:
                                acc = (tp + tn)/(tp + tn + fp + fn)
                            except:
                                acc = 0
                            try:
                                f1s = 2*(prec*rec)/(prec + rec)
                            except:
                                f1s = 0
                            avgs[:4] = np.array([prec,rec,acc,f1s])
                            print(('number batches %d, batch size %d, averages: loss  = %.4f, accuracy = %.4f, precision = %.4f, recall = %.4f, F1 score = %.4f')%((tot_iter,batch_size) + tuple(avgs[[4,2,0,1,3]])))
                        else:
                            print(('average loss %.2f')%((avgs/tot_iter)[0])) 

                                
                                
    def get_exec_func(self,jdict):
        func = getattr(importlib.import_module(jdict['from']),jdict['import'])
        kwargs = jdict['kwargs']
        return func,kwargs
    
    def set_data_handler(self,imports=None):
        '''
        Factory for data handler.
        inputs:
            imports: dict with the 'import' and 'from' keys. The import statement would actually read as 
                 from imports['from'] import imports['import']
                 Note that the last argument in __import__ is actaully called fromlist. Confusing
        outputs:
            sets self._data_handler to the BinaryDataHandler class (or a overload of it)
        '''
        if imports is None:
            self._data_handler =  BinaryDataHandler
        else:
            self._data_handler = getattr(importlib.import_module(imports['from']),imports['import'])
    
    def load_data(self,inps,handler,is_train=1):
        if is_train == 1:
            sel = 0
        elif is_train == 0:
            sel = 1
        elif is_train == 2:
            sel = 2
        else:
            print('Unrecognized value for is_train',is_train)
        fname = inps['data_file'][sel]
        ddir = inps['ddir']
        handler.load_data(fname,ddir)
        return
   
    def get_place_holders(self,inp_shapes,data_type,labels_type): 
        # Input placeholders
        with tf.name_scope('input'):
            shape = [None]
            shape.extend(inp_shapes[0])
            x = tf.placeholder(data_type, shape, name='x-input')
            shape = [None]
            if len(inp_shapes) >= 2:
                shape.extend(inp_shapes[1])
            y_ = tf.placeholder(labels_type, shape, name='y-input')
            shape = [None]
            if len(inp_shapes) == 3:
                shape.extend(inp_shapes[1])
                weights = tf.placeholder(labels_type, shape, name='weights-input')

        keep_prob = tf.placeholder(tf.float32)
        is_test = tf.placeholder(tf.bool)
        if len(inp_shapes) == 3:
            return x,y_,weights,keep_prob,is_test
        else:
            return x,y_,None,keep_prob,is_test
    def parametric_relu(self,_x):
        alphas = tf.get_variable('alpha_' + str(self._relu_cnt), _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
        self._relu_cnt += 1
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg  
    
    def get_activation(self,typea): 
        if typea == 'relu':
            ret = tf.nn.relu
        elif typea == 'relu6':
            ret = tf.nn.relu6
        elif typea == 'crelu':
            ret = tf.nn.crelu
        elif typea == 'leaky_relu':
            ret = tf.nn.leaky_relu
        elif typea == 'param_relu':          
            ret = self.parametric_relu
        elif typea == 'elu':
            ret = tf.nn.elu
        elif typea == 'softplus':
            ret = tf.nn.softplus
        elif typea == 'softsign':
            ret = tf.nn.softsign
        elif typea == 'sigmoid':
            ret = tf.sigmoid
        elif typea == 'tanh':
            ret = tf.tanh
        elif typea == 'identity':
            ret = tf.identity
        #when using composite layers the activation might be define already.
        #just return it
        elif callable(typea):
            ret = typea
        else:
            print('Unrecognized activation',typea)
            raise ValueError
        return ret
    
    def fill_feed_dict(self,datain,keep_prob,is_test,data_ph,labels_ph,weights_ph,keep_prop_ph,is_test_ph):   
        data_,labels_,weights_ = datain.next_batch()
        data = data_
        labels = labels_
        weights = weights_
        if weights is None:
            feed_dict = {
                         data_ph:data,
                         labels_ph:labels,
                         keep_prop_ph:keep_prob,
                         is_test_ph:is_test
                         }
        else:
            feed_dict = {
                         data_ph:data,
                         labels_ph:labels,
                         weights_ph:weights,
                         keep_prop_ph:keep_prob,
                         is_test_ph:is_test
                         }
        return feed_dict

    def eval_error(self,err_op,sess,typel,inputs_ph,labels_ph,weights_ph,keep_prob_ph,is_test_ph,data,use_all=False):  
        #change batch size to use the whole set. Necessary when using batch normalization
        if use_all:
            saved_batch_size = data._batch_size
            data._batch_size = data._data_size
        
        steps_per_epoch = int((data._data_size/data._batch_size))
        num_samples = data._batch_size*steps_per_epoch
        #no need for the keep_prob_ph
        tot_err = 0
        loss = 0
        is_first = True
        for step in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(sess,data,1,True,inputs_ph,labels_ph,weights_ph,keep_prob_ph,is_test_ph)
            err_now =  sess.run(err_op,feed_dict=feed_dict)
            if isinstance(err_now,int):
                tot_err += err_now[0]
            else:
                if is_first:
                    is_first = False
                    tot_err = np.array(err_now[0])
                else:
                    tot_err += np.array(err_now[0])
            loss += err_now[1]
   
        #restore original value
        if use_all:
            data._batch_size = saved_batch_size
        if typel == 'sparse_cross_entropy':
            return (tot_err/num_samples,loss/num_samples) 
        elif typel == 'l2':
            #the l2_loss used for the error has a 1/2 in front
            return (np.sqrt(2*tot_err/num_samples),loss/num_samples)
        else:
            print('Unrecognized loss type',typel)
            raise ValueError
   
    def get_error(self,typel,y,y_,metric='praf1'):
        if typel == 'sparse_cross_entropy':
            y_ = y_.astype(np.int32)
            y_ = np.reshape(y_,[-1])
            if int(y.shape[1]) == 2:
                if self._class_thr > 0:
                    assert self._class_thr < 1,'Threshold must be between 0 and 1'
                    y = np.exp(y)/np.tile(np.sum(np.exp(y),1).reshape([-1,1]),[1,2])
                    ym = (y[:,1] >= self._class_thr).astype(np.int)
                else:
                    ym = np.argmax(y,1)
                tp = np.nonzero(np.logical_and(y_ == 1,ym == 1))[0].size
                tn = np.nonzero(np.logical_and(y_ == 0,ym == 0))[0].size
                fp = np.nonzero(np.logical_and(y_ == 0,ym == 1))[0].size
                fn = np.nonzero(np.logical_and(y_ == 1,ym == 0))[0].size
                if metric == 'praf1':
                    try:
                        prec = tp/(tp + fp)
                    except:
                        prec = 0
                    try:
                        rec = tp/(tp + fn)
                    except:
                        rec = 0
                    try:
                        acc = (tp + tn)/(tp + tn + fp + fn)
                    except:
                        acc = 0
                    try:
                        f1s = 2*(prec*rec)/(prec + rec)
                    except:
                        f1s = 0
                    ret = (prec, rec ,acc, f1s)
                elif metric == 'rates':
                    ret = (tp,fp,tn,fn)
                else:
                    print('Unrecognized metric',metric)
                    raise Exception
            else:
                ym = np.argmax(y,1)
                ret = (ym == y_).astype(np.int32)
                ret = np.mean(ret)
        else:
            print('Unrecognized loss type',typel)
            raise ValueError
        return ret
    
    def error(self,typel,y,y_,weights=None):
        '''
        Compute the error.
        inputs:
        typel: type of loss
        y: logit
        y_: labels
        '''
        if typel == 'sparse_cross_entropy':
            y_ = tf.cast(y_, tf.int32)
            y_ = tf.reshape(y_,[-1])
            if int(y.shape[1]) == 2:
                ym = tf.argmax(y,1)
                tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_,1),tf.equal(ym,1)),tf.float32))
                tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_,0),tf.equal(ym,0)),tf.float32))
                fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_,0),tf.equal(ym,1)),tf.float32))
                fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_,1),tf.equal(ym,0)),tf.float32))
                ret = (tp, tn ,fp, fn)
            else:
                correct = tf.nn.in_top_k(y, y_, 1)
                ret = tf.logical_not(tf.cast(correct,tf.bool))
                ret = tf.reduce_sum(tf.cast(ret,tf.int32))
        elif typel == 'l2':
            if weights is None:
                ret = tf.nn.l2_loss(y-y_)
            else:
                ret = tf.nn.l2_loss(tf.multiply(y-y_,weights))
        else:
            print('Unrecognized loss type',typel)
            raise ValueError
        return ret

    def loss(self,typel,y,y_,weights=None,add_summ=True):
        '''
        Compute the specified loss op.
        Args:
          typel: type of loss. Values "sparse_cross_entropy" or "l2"
          y: logits
          y_: labels
          add_summ: add loss summary. default True
        '''
        if typel == 'sparse_cross_entropy':
            with tf.name_scope('sparse_cross_entropy'):
                y_ = tf.cast(y_, tf.int32)
                y_ = tf.reshape(y_,[-1])
                tloss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=y_, logits=y, name='sparse_cross_entropy')
                if weights is None:
                    loss_op = tf.reduce_mean(tloss)
                else:
                    #just in case the tloss has been reshaped
                    weights = tf.reshape(weights,tloss.shape)
                    loss_op = tf.reduce_mean(tf.multiply(tloss,weights))
               
                tf.summary.scalar('loss', loss_op)
        elif typel == 'l2':
            with tf.name_scope('l2loss'):
                if weights is None:
                    loss_op = 2*tf.nn.l2_loss(y-y_)/tf.cast(tf.size(y),tf.float32)
                else:
                    loss_op = 2*tf.nn.l2_loss(tf.multiply(y-y_,weights))/tf.cast(tf.size(y),tf.float32)
                tf.summary.scalar('loss', loss_op)
        else:
            print('Unrecognized loss type',typel)
            raise ValueError
        tf.add_to_collection('losses', loss_op)
        return tf.add_n(tf.get_collection('losses'),name='total_loss')

    def training(self,opt,loss,*args,**kwargs):
        """Sets up the training Ops.

        Creates a summarizer to track the loss over time in TensorBoard.
        
        Creates an optimizer and applies the gradients to all trainable variables.
        
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        
        Args:
          opt: type of optimizer
          loss: Loss tensor, from loss().
        Returns:
          train_op: The Op for training.
        """
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = self.get_optimizer(opt,*args,**kwargs)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def get_optimizer(self,optimizer,*args,**kwargs):
        if optimizer == 'GradientDescent':
            opt = tf.train.GradientDescentOptimizer(*args,**kwargs)
        elif optimizer == 'Adadelta':
            opt = tf.train.AdadeltaOptimizer(*args,**kwargs)
        elif optimizer == 'Adagrad':
            opt = tf.train.AdagradOptimizer(*args,**kwargs)
        elif optimizer == 'AdagradDA':
            opt = tf.train.AdagradDAOptimizer(*args,**kwargs)
        elif optimizer == 'Adam':
            opt = tf.train.AdamOptimizer(*args,**kwargs)
        elif optimizer == 'Momentum':
            opt = tf.train.MomentumOptimizer(*args,**kwargs)
        elif optimizer == 'RMSProp':
            opt = tf.train.RMSPropOptimizer(*args,**kwargs)
        else:
            print('Unrecognized optimizer',optimizer)
            raise ValueError
        return opt
       
    def get_layer_args(self,x,inps,is_test=None,keep_prob=None):
        #some common kwargs
        typel = inps['type']
        kwargs = {}
        if 'activation' in inps:
            kwargs['act'] = self.get_activation(inps['activation'])
        for k in ['apply_bn','add_summ','summ_list','stddev','wd','bias']:
            if k in inps:
                kwargs[k] = inps[k]
        if typel == 'full_conn':
            input_dim = int(x.get_shape()[1])
            output_dim = inps['size']
            args = (x,input_dim, output_dim,typel,is_test)               
        elif typel.count('conv') or typel.count('pool'):
            if typel.count('conv'):
                ksize = self.get_kernel_size(x,inps['size'])
                if typel.count('only'):
                    if 'add_bias' in inps:
                        kwargs['add_bias'] = inps['add_bias']
                    args = (x,ksize,typel)
                elif typel.count('transpose'):
                    args = (x,ksize,inps['output_shape'],typel,is_test)
                else:
                    args = (x,ksize,typel,is_test)
            else:
                ksize = inps['size']
                typep = inps['typep']
                args = (x,ksize,typel,typep)
            if 'stride' in inps:
                kwargs['stride'] = inps['stride']
            if 'padding' in inps:
                kwargs['padding'] = inps['padding']
        elif typel == 'addition':
            args = (x[0],x[1],typel,is_test)
        elif typel == 'residual':
            layers = inps['layers']
            args = (x,layers,typel,is_test,keep_prob)
            if 'repeat' in inps:
                kwargs['repeat'] = inps['repeat']
        elif typel == 'dropout':
            args = (x,keep_prob)
        elif typel == 'lrn':
            args = (x,typel)
            for k in ['depth_radius','bias','alpha','beta']:
                if k in inps:
                    kwargs[k] = inps[k]
        elif typel == 'reshape':
            args = (x,inps['shape'],typel)
        elif typel == 'batch_norm':
            args = (x,typel,is_test)
        elif typel.count('dense') and typel.count('block'):
            if 'bottleneck' in inps:
                kwargs['bottleneck'] = inps['bottleneck']
            if 'stride' in inps:
                kwargs['stride'] = inps['stride']
            if 'padding' in inps:
                kwargs['padding'] = inps['padding']
            if typel == 'dense_block':
                repeat = inps['repeat']
                args = (x,inps['size'],repeat,typel,is_test)
            else:
                args = (x,inps['size'],typel,is_test)
        elif typel == 'dense_transition':
            if 'size' in inps:
                kwargs['size'] = inps['size']
            if 'stride' in inps:
                kwargs['stride'] = inps['stride']
            if 'theta' in inps:
                kwargs['theta'] = inps['theta']
            args = (x,typel,is_test)
        elif typel == 'resize_image':
            args = (x,inps['size'],typel)
            if 'method' in inps:
                kwargs['method'] = inps['method']
        else:
            print('Unrecognized layer',typel)
            raise ValueError
        return args,kwargs
       
    def inference(self,inps,x,is_test,keep_prob):
        layers = inps['layers']
        layer = x
        do_reshape = True
        for l in layers:
            if l['type'] == 'addition' and 'names' in l:
                layer = [self._named_layers[l['names'][0]],self._named_layers[l['names'][1]]]
            '''
            if l['type'] == 'full_conn' and do_reshape:
                do_reshape = False
                dim = layer.get_shape()
                nsize = 1
                for d in dim[1:]:
                    nsize *= int(d)
                layer = tf.reshape(layer,[-1,nsize])
            '''
            layer = self.get_layer(layer,l,is_test,keep_prob)
            if l['type'] in self._inference_ops_cnt_dict:
                cnt = self._inference_ops_cnt_dict[l['type']] + 1
            else:
                cnt = 0
            self._inference_ops_cnt_dict[l['type']] = cnt
            self._inference_ops_dict[l['type'] + '_' + str(cnt)] = layer
            self._inference_ops.append(layer)
            if 'name' in l:
                self._named_layers[l['name']] = layer
           
        return layer         
     
    def get_layer(self,x,inps,is_test=None,keep_prob=None):
        typel = inps['type']
        args,kwargs = self.get_layer_args(x,inps,is_test,keep_prob)
        if typel == 'full_conn':
            layer = self.full_conn(*args,**kwargs)
        elif typel == 'max_pool1d':
            layer = self.max_pool1d(*args,**kwargs)
        elif typel == 'pool2d':
            layer = self.pool2d(*args,**kwargs)
        elif typel == 'conv1d':
            layer = self.conv1d(*args,**kwargs)
        elif typel == 'conv2d':
            layer = self.conv2d(*args,**kwargs)
        elif typel == 'conv2d_only':
            layer = self.conv2d_only(*args,**kwargs)
        elif typel == 'conv2d_transpose':
            layer = self.conv2d_transpose(*args,**kwargs)
        elif typel == 'addition':
            layer = self.addition(*args,**kwargs)
        elif typel == 'residual':
            layer = self.res_block(*args,**kwargs)
        elif typel == 'dropout':
            layer = self.dropout(*args)
        elif typel == 'lrn':
            layer = self.lrn(*args,**kwargs)       
        elif typel == 'reshape':
            layer = self.reshape(*args,**kwargs)     
        elif typel == 'batch_norm':
            layer = self.batch_norm(*args,**kwargs)
        elif typel == 'dense_block':
            layer = self.dense_block(*args,**kwargs)
        elif typel == 'dense_sub_block':
            layer = self.dense_sub_block(*args,**kwargs)
        elif typel == 'dense_transition':
            layer = self.dense_transition(*args,**kwargs)
        elif typel == 'resize_image':
            layer = self.resize_image(*args,**kwargs)
        else:
            print('Unrecognized layer',typel)
            raise ValueError
        return layer
    
    def get_kernel_size(self,x,ksize,typec='normal'):
        if len(x.get_shape().as_list()) == 3:#conv1d
            assert len(ksize) == 2,'wrong kernel size'
            ret = [ksize[0],int(x.get_shape()[2]),ksize[1]]
        elif len(x.get_shape().as_list()) == 4:#conv2d
            assert len(ksize) == 3,'wrong kernel size'
            if typec == 'normal':
                ret = [ksize[0],ksize[1],int(x.get_shape()[3]),ksize[2]]
            elif typec == 'transpose':
                ret = [ksize[0],ksize[1],ksize[2],int(x.get_shape()[3])]                       
        return ret
        
    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(self,shape,name,stddev=None,wd=None):
        """Create a weight variable with appropriate initialization."""
        if stddev is None:
            stddev = 1/shape[0]**.5
        dtype = tf.float32
        initial = tf.get_variable(name,shape,
                 initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return initial
    
    def bias_variable(self,shape,name,value=0.0):
        """Create a bias variable with appropriate initialization."""
        dtype = tf.float32
        initial =  tf.get_variable(name, shape,
                   initializer=tf.constant_initializer(value),dtype=dtype)
        return initial
    
    def variable_summaries(self,var,sum_list=None):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        if sum_list is None:
            sum_list = self._summ_list
        with tf.name_scope('summaries'):
            if 'mean' in sum_list:
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
            if 'stddev' in sum_list:
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
            if 'max' in sum_list:
                tf.summary.scalar('max', tf.reduce_max(var))
            if 'min' in sum_list:
                tf.summary.scalar('min', tf.reduce_min(var))
            if 'hist' in sum_list:
                tf.summary.histogram('hist', var)
    
    def full_conn(self,input_tensor, input_dim, output_dim, layer_name,is_test, act=tf.nn.relu,apply_bn=True,add_summ=True,summ_list=None,stddev=5e-2,wd=None,bias=0.0):
        """Reusable code for making a simple neural net layer.
    
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        print(input_tensor, input_dim, output_dim, layer_name,is_test, act,apply_bn,add_summ,summ_list,stddev,wd,bias)
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            if apply_bn:
                with tf.name_scope('weights'):
                    weights = self.weight_variable([input_dim, output_dim],'weights' + str(self._weights_cnt),stddev,wd)
                    self._weights_cnt += 1
                    self._weights_list.append(weights)
                    if add_summ:
                        self.variable_summaries(weights,summ_list)
                with tf.name_scope('WX'):
                    preactivate = tf.matmul(input_tensor, weights)
                    if add_summ:
                        self.variable_summaries(preactivate,summ_list)
                
                
                scale = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()))
                beta = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()))
                pop_mean = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()), trainable=False)
                pop_var = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()), trainable=False)
                
                bn,batch_mean,batch_var = tf.cond(is_test,
                                                  lambda:self.bn_wrap(False,preactivate,scale,beta,pop_mean,pop_var),
                                                  lambda:self.bn_wrap(True,preactivate,scale,beta,pop_mean,pop_var)
                                                  )
                '''
                bn = tf.cond(is_test,
                            lambda:tf.contrib.layers.batch_norm(preactivate,decay=0.999,center=True,scale=True,is_training=False,updates_collections=None),
                            lambda:tf.contrib.layers.batch_norm(preactivate,decay=0.999,center=True,scale=True,is_training=True,updates_collections=None)
                            )
                '''
                activations = act(bn, name='activation')
                if add_summ:
                    with tf.name_scope('activation'):
                        self.variable_summaries(activations,summ_list)
                        if apply_bn:
                            self.variable_summaries(pop_mean, ['hist'])
                            self.variable_summaries(pop_var, ['hist'])
                            self.variable_summaries(batch_mean, ['hist'])
                            self.variable_summaries(batch_var, ['hist'])           
            else:
                with tf.name_scope('weights'):
                    weights = self.weight_variable([input_dim, output_dim],'weights' + str(self._weights_cnt),stddev,wd)
                    self._weights_cnt += 1
                    self._weights_list.append(weights)
                    if add_summ:
                        self.variable_summaries(weights,summ_list)
                with tf.name_scope('biases'):
                    biases = self.bias_variable([output_dim],'biases' + str(self._biases_cnt),bias)
                    self._biases_cnt += 1
                    if add_summ:
                        self.variable_summaries(biases,summ_list)
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, weights) + biases
                    self.variable_summaries(preactivate,summ_list)
                with tf.name_scope('activation'):        
                    activations = act(preactivate)
                if add_summ:
                    with tf.name_scope('activation'):
                        self.variable_summaries(activations,summ_list)
            return activations
    
    def lrn(self,inp,layer_name,depth_radius=None,bias=None,alpha=None,beta=None):
        print(inp,layer_name,depth_radius,bias,alpha,beta)
        return tf.nn.lrn(inp,depth_radius,bias,alpha,beta,layer_name)
    
    def max_pool1d(self,inputs,ksize,layer_name,stride=2,padding='SAME',add_summ=True,summ_list=None):
        assert isinstance(ksize,int), 'Expected integer for kernel size in max_pool1d'
        assert isinstance(ksize,int), 'Expected integer for stride in max_pool1d'

        with tf.name_scope(layer_name):
            output =  tf.layers.max_pooling1d(inputs,ksize,
                        stride,padding=padding)
            if add_summ:
                self.variable_summaries(output,summ_list)
        return output
        
    def pool2d(self,inputs,ksize,layer_name,typep,stride=2,padding='SAME',add_summ=True,summ_list=None):
        print(inputs,ksize,layer_name,stride,padding,add_summ,summ_list)
        with tf.name_scope(layer_name):
            if isinstance(stride,int):
                stride = [stride,stride]
            if isinstance(ksize,int):
                ksize = [ksize,ksize]
            
            assert len(ksize) == 2, 'Expected 2 element array or tuple for kernel size in pool2d'
            assert len(stride) == 2, 'Expected 2 element array or tuple for stride  in pool2d'
            if typep == 'max':
                output =  tf.layers.max_pooling2d(inputs, ksize,
                        stride, padding=padding)
            elif typep == 'avg':
                output =  tf.layers.average_pooling2d(inputs, ksize,
                        stride, padding=padding)
            else:
                print('Error. Unrecognized pool type',typep)
                raise Exception
            if add_summ:
                self.variable_summaries(output,summ_list)
        return output
    
    def reshape(self,inputs,shape,layer_name):
        #if empty list reshape as batch_size,numel
        if len(shape) == 0:
            oshape = inputs.get_shape().as_list()
            shape = [-1,int(np.prod(np.array(inputs.get_shape().as_list()[1:])))]
            
        return tf.reshape(inputs,shape,layer_name)
    
    def conv1d(self,inputs,kernel_size,layer_name,is_test,stride=1,padding='SAME',act=tf.nn.relu,apply_bn=True,add_summ=True,summ_list=None,stddev=5e-2,wd=None,bias=0.0):
        '''
        Wrapper to nn.conv1d.
        inputs: 3D tensor [batch_size,width,in_channels]
        kernel_size: list [kernel_width,in_channels,out_channels]
        layer_name: name used for the scope.
        stride: int, number of coulums the filter is moved to the right. Default 1
        padding: the padding scheme "SAME" or "VALID". Default "SAME"
        act: activation function. Default relu
        add_summ: bool, add the summary in the weights,biases and activation. Default True
        summ_list: list of string with the possible summaries. Default in self._summ_list
                  Possible values "mean","stddev","hist","min","max".
        stddev: standard deviation to initialize weights from normal distribution.
        '''
        with tf.name_scope(layer_name):
            with tf.name_scope('kernel'):
                kernel = self.weight_variable(kernel_size,'weights' + str(self._weights_cnt), stddev,wd)
                self._weights_cnt += 1
                self._weights_list.append(kernel)
                if add_summ:
                    self.variable_summaries(kernel,summ_list)
            preactivate = tf.nn.conv1d(inputs,kernel,stride,padding.upper())
            if apply_bn:
                scale = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()))
                beta = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()))
                pop_mean = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()), trainable=False)
                pop_var = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()), trainable=False)
                
                bn,batch_mean,batch_var = tf.cond(is_test,
                                                  lambda:self.bn_wrap(False,preactivate,scale,beta,pop_mean,pop_var),
                                                  lambda:self.bn_wrap(True,preactivate,scale,beta,pop_mean,pop_var)
                                                  )
                '''
                bn = tf.cond(is_test,
                            lambda:tf.contrib.layers.batch_norm(preactivate,decay=0.999,center=True,scale=True,is_training=False,updates_collections=None),
                            lambda:tf.contrib.layers.batch_norm(preactivate,decay=0.999,center=True,scale=True,is_training=True,updates_collections=None)
                            )
                '''
                with tf.name_scope('conv'):
                    conv1 = act(bn)
                    
            else:
                with tf.name_scope('biases'):
                    biases = self.bias_variable([kernel_size[-1]],'biases'+ str(self._biases_cnt),bias)
                    self._biases_cnt += 1
                    if add_summ:
                        self.variable_summaries(biases,summ_list)
                pre_activation = tf.nn.bias_add(preactivate, biases)
                with tf.name_scope('conv'):
                    conv1 = act(pre_activation)
            if add_summ:
                self.variable_summaries(conv1,summ_list)
                if apply_bn:
                    self.variable_summaries(pop_mean, ['hist'])
                    self.variable_summaries(pop_var, ['hist'])
                    self.variable_summaries(batch_mean, ['hist'])
                    self.variable_summaries(batch_var, ['hist'])
        return conv1
    
    def conv2d(self,inputs,kernel_size,layer_name,is_test,stride=1,padding='SAME',act=tf.nn.relu,apply_bn=True,add_summ=True,summ_list=None,stddev=5e-2,wd=None,bias=0.0):
        '''
        Wrapper to nn.conv2d.
        inputs: 4D tensor [batch_size,height,width,in_channels]
        kernel_size: list [kernel_height,kernel_width,in_channels,out_channels]
        layer_name: name used for the scope.
        is_test: bool True if testing False training.
        stride: list int, number of positions the filter is shifted in each dimension of the input. Normally first 
                and last dimentions are moved by one. Default [1,1,1,1]. To use default just set to 1
        padding: the padding scheme "SAME" or "VALID". Default "SAME"
        act: activation function. Default relu.
        apply_bn: bool True if apply batch norm.
        add_summ: bool, add the summary in the weights,biases and activation. Default True
        summ_list: list of string with the possible summaries. Default in self._summ_list
                  Possible values "mean","stddev","hist","min","max".
        stddev: standard deviation to initialize weights from normal distribution.
        '''
        print(inputs,kernel_size,layer_name,is_test,stride,padding, act,apply_bn,add_summ,summ_list,stddev,wd,bias)
        try:
            len(stride)
        except Exception:
            stride = [1,stride,stride,1]
        with tf.name_scope(layer_name):
            with tf.name_scope('kernel'):
                kernel = self.weight_variable(kernel_size,'weights' + str(self._weights_cnt), stddev,wd)
                self._weights_cnt += 1
                self._weights_list.append(kernel)
                if add_summ:
                    self.variable_summaries(kernel,summ_list)
            
            preactivate = tf.nn.conv2d(inputs,kernel,stride,padding.upper())
            if apply_bn:
                scale = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()))
                beta = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()))
                pop_mean = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()), trainable=False)
                pop_var = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()), trainable=False)
                
                bn,batch_mean,batch_var = tf.cond(is_test,
                                                  lambda:self.bn_wrap(False,preactivate,scale,beta,pop_mean,pop_var),
                                                  lambda:self.bn_wrap(True,preactivate,scale,beta,pop_mean,pop_var)
                                                  )              
                '''
                bn = tf.cond(is_test,
                            lambda:tf.contrib.layers.batch_norm(preactivate,decay=0.999,center=True,scale=True,is_training=False,updates_collections=None),
                            lambda:tf.contrib.layers.batch_norm(preactivate,decay=0.999,center=True,scale=True,is_training=True,updates_collections=None)
                            )
                '''
                with tf.name_scope('conv'):
                    conv2 = act(bn)
            else:
                with tf.name_scope('biases'):
                    biases = self.bias_variable([kernel_size[-1]],'biases'+ str(self._biases_cnt),bias)
                    self._biases_cnt += 1
                    if add_summ:
                        self.variable_summaries(biases,summ_list)
                pre_activation = tf.nn.bias_add(preactivate, biases)
                with tf.name_scope('conv'):
                    conv2 = act(pre_activation)
                
            if add_summ:
                self.variable_summaries(conv2,summ_list)
                if apply_bn:
                    self.variable_summaries(pop_mean, ['hist'])
                    self.variable_summaries(pop_var, ['hist'])
                    self.variable_summaries(batch_mean, ['hist'])
                    self.variable_summaries(batch_var, ['hist'])

        return conv2
    
    def conv2d_only(self,inputs,kernel_size,layer_name,add_bias=True,stride=1,padding='SAME',act=tf.identity,add_summ=True,summ_list=None,stddev=5e-2,wd=None,bias=0.0):
        '''
        Wrapper to nn.conv2d.
        inputs: 4D tensor [batch_size,height,width,in_channels]
        kernel_size: list [kernel_height,kernel_width,in_channels,out_channels]
        layer_name: name used for the scope.
        stride: list int, number of positions the filter is shifted in each dimension of the input. Normally first 
                and last dimentions are moved by one. Default [1,1,1,1]. To use default just set to 1
        padding: the padding scheme "SAME" or "VALID". Default "SAME"
        act: activation function. Default identoity.
        add_summ: bool, add the summary in the weights,biases and activation. Default True
        summ_list: list of string with the possible summaries. Default in self._summ_list
                  Possible values "mean","stddev","hist","min","max".
        stddev: standard deviation to initialize weights from normal distribution.
        '''
        print(inputs,kernel_size,layer_name,add_bias,stride,padding, act,add_summ,summ_list,stddev,wd,bias)
        try:
            len(stride)
        except Exception:
            stride = [1,stride,stride,1]
        with tf.name_scope(layer_name):
            with tf.name_scope('kernel'):
                kernel = self.weight_variable(kernel_size,'weights' + str(self._weights_cnt), stddev,wd)
                self._weights_cnt += 1
                self._weights_list.append(kernel)
                if add_summ:
                    self.variable_summaries(kernel,summ_list)
            
            preactivate = tf.nn.conv2d(inputs,kernel,stride,padding.upper())
            if add_bias:
                with tf.name_scope('biases'):
                    biases = self.bias_variable([kernel_size[-1]],'biases'+ str(self._biases_cnt),bias)
                    self._biases_cnt += 1
                    if add_summ:
                        self.variable_summaries(biases,summ_list)
                pre_activation = tf.nn.bias_add(preactivate, biases)
            with tf.name_scope('conv_only'):
                conv2 = act(pre_activation)
                
            if add_summ:
                self.variable_summaries(conv2,summ_list)
                
        return conv2
    
    def resize_image(self,inputs,size,layer_name,method='bilinear'):
        """
        Resize image:
        Args:
        inputs: 4D tensor of [batch_size,height,width,num_channels] or
                3D tensor of [height,width,num_channels].
        size: 1D array [new_height,new_width].
        layer_name: string, name of the layer.
        method: string, the resampling method. Default: bilinear.
        returns:
        New resized image of size [batch_size,new_height,new_width,num_channels] if inputs
        is a 4D tensor [new_height,new_width,num_channels] if inputs is a 3D tensor.
                
        """
        print(inputs,size,layer_name,method)
        if method == 'bilinear':
            met = tf.image.ResizeMethod.BILINEAR
        elif method == 'bicubic':
            met = tf.image.ResizeMethod.BICUBIC
        elif method == 'nearest_neighbor':
            met = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif method == 'area':
            met = tf.image.ResizeMethod.AREA
        else:
            print('Unrecoginzed method',method)
            print('Possible values are:')
            for i in []:
                print(i)
            raise ValueError
        return tf.image.resize_images(inputs,size,met)

    def conv2d_transpose(self,inputs,kernel_size,output_shape,layer_name,is_test,stride=1,padding='SAME', act=tf.nn.relu,apply_bn=True,add_summ=True,summ_list=None,stddev=5e-2,wd=None,bias=0.0):
        '''
        Wrapper to nn.conv2d_transpose.
        inputs: 4D tensor [batch_size,height,width,in_channels]
        kernel_size: list [kernel_height,kernel_width,out_channels,in_channels]
        output_shape: A 1-D Tensor representing the output shape of the deconvolution op
        layer_name: name used for the scope.
        stride: list int, number of positions the filter is shifted in each dimension of the input. Normally first 
                and last dimentions are moved by one. Default [1,1,1,1]. To use default just set to 1
        padding: the padding scheme "SAME" or "VALID". Default "SAME"
        act: activation function. Default relu
        add_summ: bool, add the summary in the weights,biases and activation. Default True
        summ_list: list of string with the possible summaries. Default in self._summ_list
                  Possible values "mean","stddev","hist","min","max".
        stddev: standard deviation to initialize weights from normal distribution.
        '''
        print(inputs,kernel_size,output_shape,layer_name,is_test,stride,padding, act,apply_bn,add_summ,summ_list,stddev,wd,bias)
           
        try:
            len(stride)
        except Exception:
            stride = [1,stride,stride,1]
        output_shape = [inputs.get_shape().as_list()[0],output_shape[0],output_shape[1],output_shape[2]]
        with tf.name_scope(layer_name):
            with tf.name_scope('kernel'):
                kernel = self.weight_variable(kernel_size,'weights' + str(self._weights_cnt), stddev,wd)
                self._weights_cnt += 1
                self._weights_list.append(kernel)
                if add_summ:
                    self.variable_summaries(kernel,summ_list)
            
            preactivate = tf.nn.conv2d_transpose(inputs,kernel,output_shape,stride,padding.upper())
            if apply_bn:
                scale = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()))
                beta = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()))
                pop_mean = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()), trainable=False)
                pop_var = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()), trainable=False)
                
                bn,batch_mean,batch_var = tf.cond(is_test,
                                                  lambda:self.bn_wrap(False,preactivate,scale,beta,pop_mean,pop_var),
                                                  lambda:self.bn_wrap(True,preactivate,scale,beta,pop_mean,pop_var)
                                                  )              
                '''
                bn = tf.cond(is_test,
                            lambda:tf.contrib.layers.batch_norm(preactivate,decay=0.999,center=True,scale=True,is_training=False,updates_collections=None),
                            lambda:tf.contrib.layers.batch_norm(preactivate,decay=0.999,center=True,scale=True,is_training=True,updates_collections=None)
                            )
                '''
                with tf.name_scope('conv_transp'):
                    conv2 = act(bn)
            else:
                with tf.name_scope('biases'):
                    biases = self.bias_variable([kernel_size[-2]],'biases'+ str(self._biases_cnt),bias)
                    self._biases_cnt += 1
                    if add_summ:
                        self.variable_summaries(biases,summ_list)
                pre_activation = tf.nn.bias_add(preactivate, biases)
                with tf.name_scope('conv_transp'):
                    conv2 = act(pre_activation)
            if add_summ:
                self.variable_summaries(conv2,summ_list)
                if apply_bn:
                    self.variable_summaries(pop_mean, ['hist'])
                    self.variable_summaries(pop_var, ['hist'])
                    self.variable_summaries(batch_mean, ['hist'])
                    self.variable_summaries(batch_var, ['hist'])

        return conv2
    
    def contact(self,inputs,layer_name,is_test,act=tf.identity,apply_bn=False,add_summ=True,summ_list=None):
        print(inputs,layer_name,is_test,act,apply_bn,add_summ,summ_list)
        dim = len(inputs.get_shape().as_list()) - 1
        preactivate =  tf.concat(inputs,dim)
        if apply_bn:
            scale = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()))
            beta = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()))
            pop_mean = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()), trainable=False)
            pop_var = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()), trainable=False)
            
            bn,batch_mean,batch_var = tf.cond(is_test,
                                              lambda:self.bn_wrap(False,preactivate,scale,beta,pop_mean,pop_var),
                                              lambda:self.bn_wrap(True,preactivate,scale,beta,pop_mean,pop_var) 
                                              )   
            activations = act(bn, name='activation')
        else:
            activations = act(preactivate, name='activation')
        if add_summ:
            self.variable_summaries(activations,summ_list)
        return activations
    
    def addition(self,inputs,shortcut,layer_name,is_test,act=tf.identity,apply_bn=False,add_summ=True,summ_list=None):
        print(inputs,shortcut,layer_name,is_test,act,apply_bn,add_summ,summ_list)
        preactivate =  tf.add(inputs, shortcut)
        if apply_bn:
            scale = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()))
            beta = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()))
            pop_mean = tf.Variable(tf.zeros(preactivate.get_shape()[1:].as_list()), trainable=False)
            pop_var = tf.Variable(tf.ones(preactivate.get_shape()[1:].as_list()), trainable=False)
            
            bn,batch_mean,batch_var = tf.cond(is_test,
                                              lambda:self.bn_wrap(False,preactivate,scale,beta,pop_mean,pop_var),
                                              lambda:self.bn_wrap(True,preactivate,scale,beta,pop_mean,pop_var) 
                                              )   
            activations = act(bn, name='activation')
        else:
            activations = act(preactivate, name='activation')
        if add_summ:
            self.variable_summaries(activations,summ_list)
        return activations
    
    def batch_norm(self,inputs,layer_name,is_test,act=tf.identity,add_summ=True,summ_list=None):
        print(inputs,layer_name,is_test,act,add_summ,summ_list)
        scale = tf.Variable(tf.ones(inputs.get_shape()[1:].as_list()))
        beta = tf.Variable(tf.zeros(inputs.get_shape()[1:].as_list()))
        pop_mean = tf.Variable(tf.zeros(inputs.get_shape()[1:].as_list()), trainable=False)
        pop_var = tf.Variable(tf.ones(inputs.get_shape()[1:].as_list()), trainable=False)
        
        bn,batch_mean,batch_var = tf.cond(is_test,
                                          lambda:self.bn_wrap(False,inputs,scale,beta,pop_mean,pop_var),
                                          lambda:self.bn_wrap(True,inputs,scale,beta,pop_mean,pop_var)
                                          )              
        with tf.name_scope('batch_norm'):
            batch_n = act(bn)
        if add_summ:
                self.variable_summaries(batch_n,summ_list)
                self.variable_summaries(pop_mean, ['hist'])
                self.variable_summaries(pop_var, ['hist'])
                self.variable_summaries(batch_mean, ['hist'])
                self.variable_summaries(batch_var, ['hist'])
        return batch_n
    def bn_wrap(self,is_train,preactivate,scale,beta,pop_mean,pop_var,decay=0.999):
        batch_mean, batch_var = tf.nn.moments(preactivate,[0])
        #http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

        if is_train:
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                bn =  tf.nn.batch_normalization(preactivate,
                    batch_mean, batch_var, beta, scale, 1e-3)
        else:
            bn = tf.nn.batch_normalization(preactivate,
                pop_mean, pop_var, beta, scale, 1e-3)
        return bn,batch_mean,batch_var
   
    def dropout(self,layer,keep_prob):
        with tf.name_scope('dropout'):
            dropped = tf.nn.dropout(layer, keep_prob)
        return dropped
    
    def dense_transition(self,inputs,layer_name,is_test,size=2,stride=2,theta=1,typep='avg',add_summ=True,summ_list=None):
        if abs(theta - 1) > 0.0001:
            resize = int(round(inputs.get_shape().as_list()[-1]*theta))
            layer_info = {"type":"conv2d","size":[1,1,resize],"activation":tf.identity,"padding":"same","apply_bn":False,"add_summ":add_summ,"summ_list":summ_list}
            layer = self.get_layer(inputs,layer_info,is_test)
        else:
            layer= inputs
        layer_info = {"type":"pool2d","typep":typep,"size":size,"padding":"same","add_summ":add_summ,"summ_list":summ_list}
        layer = self.get_layer(layer,layer_info,is_test)
        return layer
    
    def dense_block(self,inputs_now,size,repeat,layer_name,is_test,stride=1,padding='SAME',bottleneck=None,act=tf.nn.relu,add_summ=True,summ_list=None):
        first_time = True
        ret = inputs_now
        for _ in range(repeat):
            ret = self.dense_sub_block(ret,size,layer_name,is_test,stride,padding,bottleneck,act,add_summ,summ_list)
        return ret
    
    def dense_sub_block(self,inputs_now,size,layer_name,is_test,stride=1,padding='SAME',bottleneck=None,act=tf.nn.relu,add_summ=True,summ_list=None):
        if bottleneck is not None:
            layer_info = {"type":"batch_norm","activation":act,"add_summ":add_summ,"summ_list":summ_list}
            layer = self.get_layer(inputs_now,layer_info,is_test)
            layer_info = {"type":"conv2d","size":[1,1,bottleneck],"activation":tf.identity,"padding":padding,"stride":stride,"apply_bn":False,"add_summ":add_summ,"summ_list":summ_list}
            layer = self.get_layer(layer,layer_info,is_test)
        else:
            layer = inputs_now    
        layer_info = {"type":"batch_norm","activation":act,"add_summ":add_summ,"summ_list":summ_list}
        layer = self.get_layer(layer,layer_info,is_test)
        layer_info = {"type":"conv2d","size":size,"activation":tf.identity,"padding":padding,"stride":stride,"apply_bn":False,"add_summ":add_summ,"summ_list":summ_list}
        layer = self.get_layer(layer,layer_info,is_test)
        return tf.concat([inputs_now,layer],3)
            
    def res_block(self,inputs_now,layers,layer_name,is_test,repeat=1,add_summ=True,summ_list=None):
        ret = None
        for i in range(repeat):
            inputs = inputs_now if i == 0 else ret
            ret = self.res_sub_block(inputs,layers,layer_name,is_test,add_summ,summ_list)
        return ret
    
    def res_sub_block(self,inputs_now,layers,layer_name,is_test,add_summ=True,summ_list=None):
        first_time = True
        ret = None
        for l in layers:
            if l['type'] == 'shortcut':
                inp_shape = inputs.shape.as_list()
                shct_shape = inputs_now.shape.as_list()
                #check if number of feature maps is the same,
                #if not adapt the shortcut
                if inp_shape[3] != shct_shape[3]:
                    layer_info = {"type":"conv2d","size":[1,1,inp_shape[3]],"activation":"identity","padding":"same"}
                    shortcut = self.get_layer(inputs_now,layer_info,is_test)
                    if l['type'] in self._inference_ops_cnt_dict:
                        cnt = self._inference_ops_cnt_dict[l['type']] + 1
                    else:
                        cnt = 0
                    self._inference_ops_cnt_dict[l['type']] = cnt
                    self._inference_ops_dict[l['type'] + '_' + str(cnt)] = shortcut
                    self._inference_ops.append(shortcut)
                    if 'name' in l:
                        self._named_layers[l['name']] = shortcut
                else:
                    shortcut = inputs_now
            elif l['type'] == 'addition':
                ret = self.get_layer([inputs,shortcut],l,is_test)
                if l['type'] in self._inference_ops_cnt_dict:
                    cnt = self._inference_ops_cnt_dict[l['type']] + 1
                else:
                    cnt = 0
                self._inference_ops_cnt_dict[l['type']] = cnt
                self._inference_ops_dict[l['type'] + '_' + str(cnt)] = ret
                self._inference_ops.append(ret)
                if 'name' in l:
                    self._named_layers[l['name']] = ret
            else:
                if first_time:
                    inputs = self.get_layer(inputs_now,l,is_test)
                    first_time = False
                else:
                    inputs = self.get_layer(inputs,l,is_test)
                if l['type'] in self._inference_ops_cnt_dict:
                    cnt = self._inference_ops_cnt_dict[l['type']] + 1
                else:
                    cnt = 0
                self._inference_ops_cnt_dict[l['type']] = cnt
                self._inference_ops_dict[l['type'] + '_' + str(cnt)] = inputs
                self._inference_ops.append(inputs)
                if 'name' in l:
                    self._named_layers[l['name']] = inputs
            
        return ret        
    
    def update_ops_dict(self,inputs,l):
        if l['type'] in self._inference_ops_cnt_dict:
            cnt = self._inference_ops_cnt_dict[l['type']] + 1
        else:
            cnt = 0
        self._inference_ops_cnt_dict[l['type']] = cnt
        self._inference_ops_dict[l['type'] + '_' + str(cnt)] = inputs
        self._inference_ops.append(inputs)
        if 'name' in l:
            self._named_layers[l['name']] = inputs
        return
            
