'''
Main module for cycle-gan.
'''

## built-in and 3rd party module
import time
import os
import tensorflow as tf
import numpy as np
import glob
import nibabel
from scipy.misc import imsave, imread
## local module
import cycleGAN_model as model

## parameters specific to certain data
DATAPATH = "\\\\hnascifs01.uwhis.hosp.wisc.edu\\einas01\\Groups\\" +  \
       "PETMR\\deepMRAC_pelvis\\training_data20170927_resliced_augmented_corrected_nii"
A_NAME = "axt2-???.nii"
B_NAME = "mask-???.nii"
HEIGHT = 168
WIDTH = 256
DATALENGTH = 1000

## model parameters
EPOCH = 210
BATCHSIZE = 4
LOOPS = DATALENGTH // BATCHSIZE  ## iterations within each epoch
From_ckpt = 0     ## if change this to none-0, then initialize model using checkpoint file
LEARNING_RATE = 0.0002   ## learning rate is decreasing after 100 epochs, see train()
L1_lambda = 10    ## used in generator loss, generally need not change it
POOLSIZE = 50     ## a technical part of the model, not quite important
IMGtoPATH = "./cyclegan_output_image_{}/"
CheckPointPATH = "./cyclegan_ckpt/my_model.ckpt"


def train():
    '''
    Train and evaluate the model.
    True_A, True_B are two sets of images, e.g. MR and CT images respectively.
    '''
    tf.set_random_seed(100)
    
    true_A = tf.placeholder(tf.float32, [None,HEIGHT,WIDTH,1])
    true_B = tf.placeholder(tf.float32, [None,HEIGHT,WIDTH,1])
    
    fake_B = model.Gen(true_A, name="Gen_A2B")
    fake_A = model.Gen(true_B, name="Gen_B2A")
    cyc_A = model.Gen(fake_B, name="Gen_B2A")
    cyc_B = model.Gen(fake_A, name="Gen_A2B")
    
    DB_fake = model.Disc(fake_B, name="DisB")
    DA_fake = model.Disc(fake_A, name="DisA")
    DB_real = model.Disc(true_B, name="DisB")
    DA_real = model.Disc(true_A, name="DisA")

    ## calculating generator loss
    g_loss = model.mae_criterion(DA_fake, tf.ones_like(DA_fake)) \
             + model.mae_criterion(DB_fake, tf.ones_like(DB_fake)) \
             + L1_lambda*model.abs_criterion(true_A, cyc_A) \
             + L1_lambda*model.abs_criterion(true_B, cyc_B)
    
    fake_A_sample = tf.placeholder(tf.float32,[None,HEIGHT,WIDTH,1])
    fake_B_sample = tf.placeholder(tf.float32,[None,HEIGHT,WIDTH,1])
    DA_fake_sample = model.Disc(fake_A_sample, name="DisA")
    DB_fake_sample = model.Disc(fake_B_sample, name="DisB")

    ## calculating discriminator loss
    db_loss_real = model.mae_criterion(DB_real, tf.ones_like(DB_real))
    db_loss_fake = model.mae_criterion(DB_fake_sample, tf.zeros_like(DB_fake_sample))
    db_loss = (db_loss_real + db_loss_fake) / 2
    da_loss_real = model.mae_criterion(DA_real, tf.ones_like(DA_real))
    da_loss_fake = model.mae_criterion(DA_fake_sample, tf.zeros_like(DA_fake_sample))
    da_loss = (da_loss_real + da_loss_fake) / 2
    d_loss = da_loss + db_loss
    
    fake_pool = []
    def pool(A,B):
        if len(fake_pool) < POOLSIZE:
            fake_pool.append((A,B))
            return A,B
        else:
            if np.random.rand() > 0.5:
                return A,B
            else:
                dice2 = np.random.randint(50)
                temp = fake_pool[dice2]
                fake_pool[dice2] = (A,B)
                return temp[0], temp[1]


    train_vars = tf.trainable_variables()
    disc_vars = [var for var in train_vars if "Dis" in var.name]
    gen_vars = [var for var in train_vars if "Gen" in var.name]

    lr = tf.placeholder(tf.float32, name="learning_rate")
    optimizer = tf.train.AdamOptimizer(lr)
    train_gen = optimizer.minimize(g_loss, var_list=gen_vars)
    train_disc = optimizer.minimize(d_loss, var_list=disc_vars)

    ## load and normalize data to [-1,1]
    A_data = (prepare_data(DATAPATH,A_NAME) + 1.82) / 5.1 -1 
    B_data =  prepare_data(DATAPATH,B_NAME)  / 1.5 - 1  
    test_A_data = A_data[DATALENGTH:,:,:,:]
    test_B_data = B_data[DATALENGTH:,:,:,:]
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        if From_ckpt == 0:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess,"{}-{}".format(CheckPointPATH,From_ckpt))
            
        for epoch in range(EPOCH):
            lrate = LEARNING_RATE if epoch < 100 else LEARNING_RATE*(EPOCH - epoch)/100
            for i in range(LOOPS):
                tic = time.time()
                ## get training batch
                A_batch = A_data[(i*BATCHSIZE):((i+1)*BATCHSIZE),:]
                B_batch = B_data[(i*BATCHSIZE):((i+1)*BATCHSIZE),:]
                ## train generator and return fake images
                tempA, tempB, _ = sess.run([fake_A,fake_B,train_gen],feed_dict={
                    true_A:A_batch, true_B:B_batch, lr:lrate})
                tempA, tempB = pool(tempA, tempB)
                ## train_discriminator using generated fake images
                sess.run(train_disc,feed_dict={
                    true_A:A_batch, true_B:B_batch,
                    fake_A_sample:tempA, fake_B_sample:tempB, lr:lrate})
                print("Epoch {}: LOOP: {} Time: {:.4f}".format(epoch,i,time.time()-tic))

            if epoch % 50 == 5:
                saver.save(sess, CheckPointPATH, global_step = epoch)

            ## evaluate current model using test set, every 10 epochs 
            if epoch % 10 == 9:
                if not os.path.exists(IMGtoPATH.format(epoch)):
                    os.mkdir(IMGtoPATH.format(epoch))
                for j in range(test_A_data.shape[0]):
                    result_img = sess.run(fake_B,feed_dict={
                            true_A:test_A_data[j:(j+1),:,:,:]})[0,:,:,0]
                    concat_img = np.concatenate((test_A_data[j,:,:,0], \
                                                 result_img, \
                                                 test_B_data[j,:,:,0]),axis=0)
                    imsave(os.path.join(IMGtoPATH.format(epoch),"cycle_output{}.jpg".format(j)),concat_img)    



def prepare_data(path, name):
    '''
    For preparing the nii data into np.array
    Output: 4-dim tensor, [batchsize,height,width,1]
    Note: hard coding in height: 168. Because model requires input
          height and width are multiples of 8.
    '''
    ## there is some issue in order using glob.glob, so use sorted()
    imglist = []
    nii_names = sorted(glob.glob(os.path.join(path,name)))
    for nii_name in nii_names:
        img_slice = nibabel.load(nii_name).get_data()[1:169,:,:]
        for j in range(img_slice.shape[2]):
            if np.max(img_slice[:,:,j:(j+1)]) != np.min(img_slice[:,:,j:(j+1)]):
                imglist.append(np.array(img_slice[:,:,j:(j+1)]))
    return np.array(imglist,dtype=np.float32)

##data = prepare_data(DATAPATH,A_NAME)
##print(data.shape)  ## 1467   ## some black images
##dataB=  prepare_data(DATAPATH,B_NAME)
##print(dataB.shape)

if __name__ == "__main__":
    train()

