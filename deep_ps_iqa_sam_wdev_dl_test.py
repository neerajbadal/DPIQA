'''
Created on 21-Mar-2021

@author: shrib
'''
import tensorflow as tf
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import  Input
from tensorflow.keras.layers import Conv2D, ReLU

import numpy as np
from tensorflow.keras import regularizers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from scipy.stats import spearmanr,pearsonr
from sklearn.metrics import  mean_squared_error
import scipy.optimize as optimization



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
#     pan_feat_l2 = BatchNormalization()(pan_feat_l2)                    
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
    
    
    
    ''' creating multi-spectral layer 2'''
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



''' Defining Custom Validation Function'''
def val_step_ii(image_batch):
    ps_inp,ms_inp,pan_inp,q2n_gt = image_batch
    ms_inp = tf.image.resize(ms_inp,[256,256], method=tf.image.ResizeMethod.BICUBIC)
#     ms_inp = tf.image.resize_with_crop_or_pad(ms_inp,256,256)
    q2n_gt = tf.expand_dims(q2n_gt,1)
    q2n_pred = iqa_mod([pan_inp,ms_inp,ps_inp])
    clubbed_out = np.c_[q2n_pred,q2n_gt]
    
    return clubbed_out
               
               
save_dir = "/home/sac/dpiqa/A2_PS_Data/"

def linearFit(x, m, c):
#     logistic_term = (0.5 - (1.0/(1.0+np.exp(B*(x-C)))))
    val_ = m * x + c
    return val_
               
def train_ops_ii(dataset_val):
    for epoch in range(1):
        vae_loss_list_val = []
                       
        i_c = 0
        for image_batch in dataset_val:
            val_batch_res = val_step_ii(image_batch)
            vae_loss_list_val.extend(list(val_batch_res))
                           
                           
            i_c += 1
                       
        vae_loss_list_val = np.array(vae_loss_list_val)    
        print("dataset population",vae_loss_list_val.shape[0])
        srocc_ = spearmanr(vae_loss_list_val[:,0],vae_loss_list_val[:,1])
        
        plcc_ = pearsonr(vae_loss_list_val[:,0],vae_loss_list_val[:,1])
        rmse_ = mean_squared_error(vae_loss_list_val[:,1]
                                   ,vae_loss_list_val[:,0],squared=False)
        
        guess = [1.0,0.0]
        params, params_covariance = optimization.curve_fit(linearFit,vae_loss_list_val[:,0],
                                                            vae_loss_list_val[:,1], guess,
                                                       maxfev = 10000)
        
        rmse_mean_lfit = mean_squared_error(vae_loss_list_val[:,1],
                                            linearFit(vae_loss_list_val[:,0],*params),squared=False)
        
        
        plcc_fit = pearsonr(linearFit(vae_loss_list_val[:,0],*params),vae_loss_list_val[:,1])
        
        print("SROCC ",np.round(srocc_[0],3)," PLCC ",
              np.round(plcc_[0],3)," PLCC on linear fit ",np.round(plcc_fit[0],3)," rmse ",np.round(rmse_,5)," RMSE after linear fit ",np.round(rmse_mean_lfit,4))
                
        return  vae_loss_list_val      
               


if __name__ == "__main__":
    ''' 
    arg[0] : pansharpened_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative (from code file) paths as string of pansharpened 
             image patch filenames whose quality metric is to estimated
     
    arg[1] : multispectral_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative paths (from code file)
             as string of multispectral image patch filenames.
     
    arg[2] : panchromatic_file_names_numpy_file :
             absolute path of a numpy file containing absolute (recommended) / relative paths (from code file) as string of panchromatic 
             image patch filenames.
     
    arg[3] : gt_numpy_file : a numpy file containing array of SAM scores in the same index order as above input images.
     
    arg[4] : image_patch_absolute_dir : absolute path of directory where image patches are stored (to be prefixed for first 3 arguments). 
    
    arg[5] : saved_model_path_ : absolute path of saved model
     
    '''
    
    image_patch_absolute_dir = "./image_patches/wv2_ct2/"
    
    pansharpened_file_names_numpy_file = "./input_numpy_files/sam/test/resnet_test_gap_wv2_wdevsam_sac_dl_id.npy"
    pansharpened_test_image_file_names = np.load(pansharpened_file_names_numpy_file).astype('U')
    
    pansharpened_test_image_file_names = np.array([ image_patch_absolute_dir + f_nm 
                                                   for f_nm in pansharpened_test_image_file_names ])  
    
    multispectral_file_names_numpy_file = "./input_numpy_files/sam/test/resnet_test_gap_wv2_wdevsam_sac_dl_msinp.npy"
    multispectral_inp_test_image_file_names = np.load(multispectral_file_names_numpy_file)
    
    multispectral_inp_test_image_file_names = np.array([ image_patch_absolute_dir + f_nm 
                                                        for f_nm in multispectral_inp_test_image_file_names ])

    
    panchromatic_file_names_numpy_file = "./input_numpy_files/sam/test/resnet_test_gap_wv2_wdevsam_sac_dl_paninp.npy"
    panchromatic_inp_test_image_file_names = np.load(panchromatic_file_names_numpy_file)
    panchromatic_inp_test_image_file_names = np.array([ image_patch_absolute_dir + f_nm 
                                                        for f_nm in panchromatic_inp_test_image_file_names ])
    
    gt_numpy_file = "./input_numpy_files/sam/test/resnet_test_gap_wv2_wdevsam_sac_dl_gt.npy" 
    test_gt = np.load(gt_numpy_file)
    
    iqa_mod = ps_iqa_model([256,256,1])
    
    checkpoint = tf.train.Checkpoint(iqa_mod)
    
    saved_model_path_ = "./final_saved_models/sam_model/-112"
    
    checkpoint.restore(saved_model_path_)#CT WV3
    
    batch_size = 36
    
    test_batch_generator = My_Custom_Generator(pansharpened_test_image_file_names,
                                               multispectral_inp_test_image_file_names,
                                               panchromatic_inp_test_image_file_names,
                                               test_gt,batch_size)
    scatter_vals = train_ops_ii(test_batch_generator)
