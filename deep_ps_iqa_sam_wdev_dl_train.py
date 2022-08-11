'''
Created on 21-Mar-2021

@author: shrib
'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, ReLU

import numpy as np
from tensorflow.keras import regularizers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from scipy.stats import spearmanr
import os

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
loss_lamda = 0

''' function to create multispectral feature extraction blocks '''
def ms_feature_blocks():
    ps_inp = Input([None,None,4])
    
    ps_feat = Conv2D(16, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal',strides=2)(ps_inp)
    
    ps_feat = ReLU()(ps_feat)
    
    ps_feat = Conv2D(16, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal')(ps_feat)                    
    
    shortcut = Conv2D(filters=16,kernel_size=(1, 1),strides=2,
                                    kernel_initializer='he_normal',
                                    
                kernel_regularizer=regularizers.l2(loss_lamda)                    
                                    )(ps_inp)
    
    
    ps_feat = tf.keras.layers.add([ps_feat, shortcut])

    
    ps_feat_l1 = ReLU()(ps_feat)
    
    
    ps_feat_l2 = Conv2D(64, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal',strides=2)(ps_feat_l1)
    
    ps_feat_l2 = ReLU()(ps_feat_l2)
    
    ps_feat_l2 = Conv2D(64, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal')(ps_feat_l2)

    shortcut = Conv2D(filters=64,kernel_size=(1, 1),strides=2,
                                    kernel_initializer='he_normal',
                                    
                kernel_regularizer=regularizers.l2(loss_lamda)                    
                                    )(ps_feat_l1)
    
    
    ps_feat_l2 = tf.keras.layers.add([ps_feat_l2, shortcut])
    
    ps_feat_l2 = ReLU()(ps_feat_l2)

    
    ps_feat_l3 = Conv2D(128, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal',strides=2)(ps_feat_l2)
    ps_feat_l3 = ReLU()(ps_feat_l3)
    
    ps_feat_l3 = Conv2D(128, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal')(ps_feat_l3)
    
    shortcut = Conv2D(filters=128,kernel_size=(1, 1),strides=2,
                                    kernel_initializer='he_normal',
                                    
                kernel_regularizer=regularizers.l2(loss_lamda)                    
                                    )(ps_feat_l2)
    
    
    ps_feat_l3 = tf.keras.layers.add([ps_feat_l3, shortcut])
    
    ps_feat_l3 = ReLU()(ps_feat_l3)
    
    ps_feat_l4 = Conv2D(256, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal',strides=2)(ps_feat_l3)
    ps_feat_l4 = ReLU()(ps_feat_l4)
    
    ps_feat_l4 = Conv2D(256, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal')(ps_feat_l4)
    
    shortcut = Conv2D(filters=256,kernel_size=(1, 1),strides=2,
                                    kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(loss_lamda)                    
                                    
                                    )(ps_feat_l3)
    
    
    ps_feat_l4 = tf.keras.layers.add([ps_feat_l4, shortcut])
    
    ps_feat_l4 = ReLU()(ps_feat_l4)
    
    ms_model = Model(inputs=[ps_inp], outputs=[ps_feat_l1,ps_feat_l2,ps_feat_l3,ps_feat_l4])
    
    return ms_model

''' function to create overall DPIQA network '''
def ps_iqa_model(pan_shape):

    pan_inp = Input(pan_shape)
    ms_inp = Input([None,None,4])
    ps_inp = Input([None,None,4])
    
    ms_input_model = ms_feature_blocks()

    ms_hl_outputs = ms_input_model(ms_inp)
    ps_hl_outputs = ms_input_model(ps_inp)
    
    pan_feat = Conv2D(16, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal',strides=2)(pan_inp)
    pan_feat = ReLU()(pan_feat)
    
    pan_feat = Conv2D(16, kernel_size=3,padding='same',kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        )(pan_feat)
    
    shortcut = Conv2D(filters=16,kernel_size=(1, 1),strides=2,
                                    kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(loss_lamda)                    
                                    
                                    )(pan_inp)
    
    
    pan_feat = tf.keras.layers.add([pan_feat, shortcut])
    pan_feat_l1 = ReLU()(pan_feat)
    
    
    ''' creating multi-spectral layer 1 '''
    ms_layer_output = ms_hl_outputs[0]#ms_input_model.get_layer(layer_name).output
    
    
    pan_feat_l1 = tf.keras.layers.concatenate([pan_feat_l1,ms_layer_output], axis=3)
    
    pan_feat_l2 = Conv2D(64, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal',strides=2)(pan_feat_l1)
    pan_feat_l2 = ReLU()(pan_feat_l2)
                        
    pan_feat_l2 = Conv2D(64, kernel_size=3,padding='same',kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        )(pan_feat_l2)                    
    
    shortcut = Conv2D(filters=64,kernel_size=(1, 1),strides=2,
                                    kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(loss_lamda)                    
                                    )(pan_feat_l1)
    
    
    pan_feat_l2 = tf.keras.layers.add([pan_feat_l2, shortcut])
    pan_feat_l2 = ReLU()(pan_feat_l2)
    
    
    ''' creating multi-spectral layer 2 '''
    ms_layer_output = ms_hl_outputs[1]#ms_input_model.get_layer(layer_name).output
    pan_feat_l2 = tf.keras.layers.concatenate([pan_feat_l2,ms_layer_output], axis=3)
    
    pan_feat_l3 = Conv2D(128, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal',strides=2)(pan_feat_l2)
    pan_feat_l3 = ReLU()(pan_feat_l3)
    
    pan_feat_l3 = Conv2D(128, kernel_size=3,padding='same',kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        )(pan_feat_l3)                    
    
    shortcut = Conv2D(filters=128,kernel_size=(1, 1),strides=2,
                                    kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(loss_lamda)            
                                    )(pan_feat_l2)
    
    
    pan_feat_l3 = tf.keras.layers.add([pan_feat_l3, shortcut])

    pan_feat_l3 = ReLU()(pan_feat_l3)
    
    ''' creating multi-spectral layer 3'''
    ms_layer_output = ms_hl_outputs[2]#ms_input_model.get_layer(layer_name).output
    pan_feat_l3 = tf.keras.layers.concatenate([pan_feat_l3,ms_layer_output], axis=3)
    
    pan_feat_l4 = Conv2D(256, kernel_size=3,padding='same',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        ,kernel_initializer='he_normal',strides=2)(pan_feat_l3)
    pan_feat_l4 = ReLU()(pan_feat_l4)
                        
    pan_feat_l4 = Conv2D(256, kernel_size=3,padding='same',kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(loss_lamda)
                        )(pan_feat_l4)
 
    shortcut = Conv2D(filters=256,kernel_size=(1, 1),strides=2,
                                    kernel_initializer='he_normal'
                                    ,kernel_regularizer=regularizers.l2(loss_lamda)
                                    )(pan_feat_l3)
    
    
    pan_feat_l4 = tf.keras.layers.add([pan_feat_l4, shortcut])

    pan_feat_l4 = ReLU()(pan_feat_l4)
    
    pseudo_ps_vector = tf.reshape(pan_feat_l4,shape=[-1,tf.keras.backend.int_shape(pan_feat_l4)[1]*
                                                   tf.keras.backend.int_shape(pan_feat_l4)[2],
                                                   tf.keras.backend.int_shape(pan_feat_l4)[3]])
     
    
    
    pseudo_ps_vector = tf.math.l2_normalize(pseudo_ps_vector, axis=2, epsilon=1e-12, name=None)
  
        
    ps_feat = ps_hl_outputs[3]
    ps_feat = tf.reshape(ps_feat,shape=[-1,tf.shape(ps_feat)[1]*
                                                   tf.shape(ps_feat)[2],
                                                   tf.shape(ps_feat)[3]])
     
    ps_feat = tf.math.l2_normalize(ps_feat, axis=2, epsilon=1e-12, name=None)
     
     
     
    similarity_vector = tf.keras.backend.sum(pseudo_ps_vector * ps_feat,axis=2,keepdims=True)
    similarity_vector = tf.reshape(similarity_vector,shape=[-1,tf.keras.backend.int_shape(similarity_vector)[1]*
                                               tf.keras.backend.int_shape(similarity_vector)[2]])
       
    
    cos_val = similarity_vector
    cos_val = tf.clip_by_value(cos_val, -1.0,1.0)
    
    sam_pred = tf.acos(cos_val)
    
    sam_pred = tf.reduce_mean(sam_pred,axis=1,keepdims=True)

    
    qa_model = Model(inputs=[pan_inp,ms_inp,ps_inp],
                             outputs=[sam_pred])
    print(qa_model.summary())
    return qa_model
       

# iqa_mod = ps_iqa_model([256,256,1])
# 
# model_optimizer = tf.keras.optimizers.Adam(4e-7,beta_1=0.6,beta_2=0.6)
# checkpoint = tf.train.Checkpoint(iqa_mod)

 

''' Defining Custom Training Function'''
from matplotlib import pyplot as plt
# @tf.function
def train_step(image_batch):
    ps_inp,ms_inp,pan_inp,q2n_gt = image_batch
    
    ms_inp = tf.image.resize(ms_inp,[256,256], method=tf.image.ResizeMethod.BICUBIC)
    
     
    q2n_gt = tf.expand_dims(q2n_gt,1)
    with tf.GradientTape() as vae_tape:
        
        q2n_pred = iqa_mod([pan_inp,ms_inp,ps_inp])
        
        srocc_ = tf.py_function(spearmanr, [q2n_pred, 
                       q2n_gt], Tout = tf.float32)
        lossL1 = tf.reduce_sum(iqa_mod.losses)
        mse_loss_passon = tf.reduce_mean(tf.keras.losses.mse(q2n_gt,q2n_pred))
        mse_loss = mse_loss_passon + lossL1
    gradients_of_vae = vae_tape.gradient(mse_loss, iqa_mod.trainable_variables)
     
    model_optimizer.apply_gradients(zip(gradients_of_vae, iqa_mod.trainable_variables))
    return [mse_loss_passon,srocc_,tf.reduce_mean(q2n_pred),tf.reduce_mean(q2n_gt)]
 
''' Defining Custom Validation Function'''
@tf.function
def val_step(image_batch):
    ps_inp,ms_inp,pan_inp,q2n_gt = image_batch
    q2n_gt = tf.expand_dims(q2n_gt,1)
    ms_inp = tf.image.resize(ms_inp,[256,256], method=tf.image.ResizeMethod.BICUBIC)
    q2n_pred = iqa_mod([pan_inp,ms_inp,ps_inp])
 
    srocc_ = tf.py_function(spearmanr, [q2n_pred, 
                       q2n_gt], Tout = tf.float32)
    
    mse_loss_val = tf.reduce_mean(tf.keras.losses.mse(q2n_gt,q2n_pred)) #+ lossL1
    return [mse_loss_val,srocc_,tf.reduce_mean(q2n_pred),tf.reduce_mean(q2n_gt)]

def train_ops(dataset_tr,dataset_val=None):
    for epoch in range(EPOCHS):
        vae_loss_list = []
        vae_loss_list_val = []
        srocc_list = []
        srocc_list_val = []
        
        i_c = 0
        for image_batch in dataset_tr:
            train_batch_res = train_step(image_batch)
            vae_loss_list.append(train_batch_res[0])
            srocc_list.append([epoch,i_c,train_batch_res[1],train_batch_res[2],train_batch_res[3]])
            i_c += 1
          
        srocc_list = np.array(srocc_list)
        srocc_bw.append(srocc_list)
        
        epoch_train_loss_mean.append(np.mean(vae_loss_list))
        print("Epoch ",epoch," train metrics : loss ",epoch_train_loss_mean[-1]," srocc ",np.mean(srocc_list[:,2]))
 
        i_c = 0
        for image_batch in dataset_val:
            val_batch_res = val_step(image_batch)
            #           val_batch_res = tf.math.reduce_mean(val_batch_res)
            vae_loss_list_val.append(val_batch_res[0])
            srocc_list_val.append([epoch,i_c,val_batch_res[1],val_batch_res[2],val_batch_res[3]])
            
            
            i_c += 1
        
        srocc_list_val = np.array(srocc_list_val)
        srocc_bw_test.append(srocc_list_val)   
        epoch_val_loss_mean.append(np.mean(vae_loss_list_val))
        
        print("Epoch ",epoch," val metrics : loss ",epoch_val_loss_mean[-1]," srocc ",np.mean(srocc_list_val[:,2]))
        if (epoch % 5 == 0 or epoch == EPOCHS-1) and epoch != 0:
            save_path = checkpoint.save(checkpoint_filepath)
        np.save(checkpoint_filepath+"/epoch_train_loss.npy",np.array(epoch_train_loss_mean))
        np.save(checkpoint_filepath+"/epoch_val_loss.npy",np.array(epoch_val_loss_mean))



#custom generator to pick file one by one
class My_Custom_Generator(tf.keras.utils.Sequence) :
    def __init__(self, image_filenames,msinp_filnames,
                 paninp_filnames,gt_,batch_size) :
        self.image_filenames = image_filenames
        self.gt_ = gt_
        
        self.feat_ = np.c_[self.image_filenames,msinp_filnames,
                 paninp_filnames,self.gt_]
        
#         np.random.shuffle(self.feat_)
        self.batch_size = batch_size
      
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    
    
    def __getitem__(self, idx) :
#         batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x0 = self.feat_[idx * self.batch_size : (idx+1) * self.batch_size,0]
        batch_x1 = self.feat_[idx * self.batch_size : (idx+1) * self.batch_size,1]
        batch_x2 = self.feat_[idx * self.batch_size : (idx+1) * self.batch_size,2]
#         batch_x3 = self.feat_[idx * self.batch_size : (idx+1) * self.batch_size,3]
        
        batch_y = self.feat_[idx * self.batch_size : (idx+1) * self.batch_size,3]
        load_batch0 = [np.load(batch_x0[id_]) for id_ in range(0,len(batch_x0))]
        load_batch1 = [np.load(batch_x1[id_]) for id_ in range(0,len(batch_x1))]
        load_batch2 = [np.load(batch_x2[id_]) for id_ in range(0,len(batch_x2))]
#         load_batch3 = [np.load(batch_x3[id_]) for id_ in range(0,len(batch_x3))]
        load_batch0 =  np.array(load_batch0)
        load_batch1 =  np.array(load_batch1)
        load_batch2 =  np.array(load_batch2)
#         load_batch3 =  np.array(load_batch3)
        load_batch2 = np.expand_dims(load_batch2,axis=3)
        return (load_batch0,load_batch1,load_batch2,batch_y.astype(float))

# save_dir = "/home/sac/dpiqa/A2_PS_Data/"
#  
# # train_filnames = np.load(save_dir+"resnet_train_gap_c2s_wdev_dl_id.npy")
# train_filnames = np.load(save_dir+"resnet_train_gap_wdevsam_sac_dl_id.npy")
#  
# train_filnames = np.array([f_nm.split('/')[-1] for f_nm in train_filnames])
# np.save("./input_numpy_files/sam/train/"+"resnet_train_gap_wdevsam_sac_dl_id.npy",train_filnames)
#  
# test_filnames = np.load(save_dir+"resnet_test_gap_wv2_wdevsam_sac_dl_id.npy")
#  
# msinp_train_filnames = np.load(save_dir+"resnet_train_gap_wdevsam_sac_dl_msinp.npy")
# msinp_train_filnames = np.array([f_nm.split('/')[-1] for f_nm in msinp_train_filnames])
# np.save("./input_numpy_files/sam/train/"+"resnet_train_gap_wdevsam_sac_dl_msinp.npy",msinp_train_filnames)
#  
# msinp_test_filnames = np.load(save_dir+"resnet_test_gap_wv2_wdevsam_sac_dl_msinp.npy")
#  
# paninp_train_filnames = np.load(save_dir+"resnet_train_gap_wdevsam_sac_dl_paninp.npy")
# paninp_train_filnames = np.array([f_nm.split('/')[-1] for f_nm in paninp_train_filnames])
# np.save("./input_numpy_files/sam/train/"+"resnet_train_gap_wdevsam_sac_dl_paninp.npy",paninp_train_filnames)
#  
# paninp_test_filnames = np.load(save_dir+"resnet_test_gap_wv2_wdevsam_sac_dl_paninp.npy")
#  
# train_gt = np.load(save_dir+"resnet_train_gap_wdevsam_sac_dl_gt.npy")
# np.save("./input_numpy_files/sam/train/"+"resnet_train_gap_wdevsam_sac_dl_gt.npy",train_gt)
#  
# test_gt = np.load(save_dir+"resnet_test_gap_wv2_wdevsam_sac_dl_gt.npy")
#  
# exit(0)

if __name__ == "__main__":
    ''' 
    arg[0] : train_pansharpened_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative (from code file) paths as string of pansharpened 
             image patch file names whose quality metric is to estimated
     
    arg[1] : train_multispectral_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative paths (from code file)
             as string of multispectral image patch filenames.
     
    arg[2] : train_panchromatic_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative paths (from code file) as string of panchromatic 
             image patch filenames.
     
    arg[3] : train_gt_numpy_file : a numpy file containing array of SAM scores in the same index order as above input images.
     
    arg[4] : train_image_patch_absolute_dir : absolute path of directory where image patches are stored (to be prefixed for first 3 arguments). 
    
    arg[5] : test_pansharpened_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative (from code file) paths as string of pansharpened 
             image patch file names whose quality metric is to estimated
     
    arg[6] : test_multispectral_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative paths (from code file)
             as string of multispectral image patch filenames.
     
    arg[7] : test_panchromatic_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative paths (from code file) as string of panchromatic 
             image patch filenames.
     
    arg[8] : test_gt_numpy_file : a numpy file containing array of SAM scores in the same index order as above input images.
     
    arg[9] : test_image_patch_absolute_dir : absolute path of directory where image patches are stored (to be prefixed for first 3 arguments).      
    
    arg[10] : save_model_path_ : absolute path of saved model
     
    '''
    
    
    ''' loading train test data set from wv3 test data '''
    
    train_image_patch_absolute_dir = "./image_patches/wv3c_ct2/"
    
    train_pansharpened_file_names_numpy_file = "./input_numpy_files/sam/train/resnet_train_gap_wdevsam_sac_dl_id.npy"
    pansharpened_train_image_file_names = np.load(train_pansharpened_file_names_numpy_file).astype('U')
    
    pansharpened_train_image_file_names = np.array([ train_image_patch_absolute_dir + f_nm 
                                                   for f_nm in pansharpened_train_image_file_names ])  
    
    train_multispectral_file_names_numpy_file = "./input_numpy_files/sam/train/resnet_train_gap_wdevsam_sac_dl_msinp.npy"
    multispectral_inp_train_image_file_names = np.load(train_multispectral_file_names_numpy_file)
    
    multispectral_inp_train_image_file_names = np.array([ train_image_patch_absolute_dir + f_nm 
                                                        for f_nm in multispectral_inp_train_image_file_names ])

    
    train_panchromatic_file_names_numpy_file = "./input_numpy_files/sam/train/resnet_train_gap_wdevsam_sac_dl_paninp.npy"
    panchromatic_inp_train_image_file_names = np.load(train_panchromatic_file_names_numpy_file)
    panchromatic_inp_train_image_file_names = np.array([ train_image_patch_absolute_dir + f_nm 
                                                        for f_nm in panchromatic_inp_train_image_file_names ])
    
    train_gt_numpy_file = "./input_numpy_files/sam/train/resnet_train_gap_wdevsam_sac_dl_gt.npy" 
    train_gt = np.load(train_gt_numpy_file)
    
    
    
    ''' loading test test data set from wv3 test data '''
    
    test_image_patch_absolute_dir = "./image_patches/wv3c_ct2/"
    
    test_pansharpened_file_names_numpy_file = "./input_numpy_files/sam/test/resnet_test_gap_wv3_wdevsam_sac_dl_id.npy"
    pansharpened_test_image_file_names = np.load(test_pansharpened_file_names_numpy_file).astype('U')
    
    pansharpened_test_image_file_names = np.array([ test_image_patch_absolute_dir + f_nm 
                                                   for f_nm in pansharpened_test_image_file_names ])  
    
    test_multispectral_file_names_numpy_file = "./input_numpy_files/sam/test/resnet_test_gap_wv3_wdevsam_sac_dl_msinp.npy"
    multispectral_inp_test_image_file_names = np.load(test_multispectral_file_names_numpy_file)
    
    multispectral_inp_test_image_file_names = np.array([ test_image_patch_absolute_dir + f_nm 
                                                        for f_nm in multispectral_inp_test_image_file_names ])

    
    test_panchromatic_file_names_numpy_file = "./input_numpy_files/sam/test/resnet_test_gap_wv3_wdevsam_sac_dl_paninp.npy"
    panchromatic_inp_test_image_file_names = np.load(test_panchromatic_file_names_numpy_file)
    panchromatic_inp_test_image_file_names = np.array([ test_image_patch_absolute_dir + f_nm 
                                                        for f_nm in panchromatic_inp_test_image_file_names ])
    
    test_gt_numpy_file = "./input_numpy_files/sam/test/resnet_test_gap_wv3_wdevsam_sac_dl_gt.npy" 
    test_gt = np.load(test_gt_numpy_file)
    
    
    
    save_model_path_ = "./under_train_models/sam_model_1/"
    
    iqa_mod = ps_iqa_model([256,256,1])
    
    model_optimizer = tf.keras.optimizers.Adam(4e-7,beta_1=0.6,beta_2=0.6)
    checkpoint = tf.train.Checkpoint(iqa_mod)
     
    checkpoint_filepath = save_model_path_
    
    
    try:
        os.makedirs(checkpoint_filepath)
    except OSError:
        pass
    print(checkpoint_filepath)
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_filepath))
    
    batch_size = 36
    epoch_train_loss_mean = []
    epoch_val_loss_mean = []
    srocc_bw = [] 
    srocc_bw_test = [] 
    EPOCHS = 400

    
    
    
    tr_batch_generator = My_Custom_Generator(pansharpened_train_image_file_names,
                                               multispectral_inp_train_image_file_names,
                                               panchromatic_inp_train_image_file_names,
                                               train_gt,batch_size)

    data_iter = lambda: (s for s in tr_batch_generator)
    
    dataset_tr = tf.data.Dataset.from_generator(
        data_iter,
        output_signature=(
            tf.TensorSpec(shape=(None, 256,256,4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 64,64,4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 256,256,1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ))
    
    
    # dataset_tr = dataset_tr.cache("/home/sac/dpiqa/cache_/tmpq2n.tfcache")
    # dataset_tr = dataset_tr.prefetch(130)
    
    test_batch_generator = My_Custom_Generator(pansharpened_test_image_file_names,
                                               multispectral_inp_test_image_file_names,
                                               panchromatic_inp_test_image_file_names,
                                               test_gt,batch_size)
    
    data_iter_test = lambda: (s for s in test_batch_generator)
    
    dataset_test = tf.data.Dataset.from_generator(
        data_iter_test,
        output_signature=(
            tf.TensorSpec(shape=(None, 256,256,4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 64,64,4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 256,256,1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ))
    
    # dataset_test = dataset_test.cache("/home/sac/dpiqa/cache_/tmptestq2n.tfcache")
    # dataset_test = dataset_test.prefetch(buffer_size=30)
    
    train_ops(dataset_tr,dataset_test)
