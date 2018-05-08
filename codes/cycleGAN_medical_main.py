## this and cycleGAN_model and only needs ops.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.misc import imsave, imread
import cycleGAN_medical_model as model
import glob
import matplotlib.pyplot as plt
import nibabel

plt.ion()

DATAPATH = "\\\\hnascifs01.uwhis.hosp.wisc.edu\\einas01\\Groups\\" +  \
       "PETMR\\deepMRAC_pelvis\\training_data20170927_resliced_augmented_corrected_nii"
IMGtoPATH = "./my_output_medi_normd_matched{}/"
CheckPointPATH = "./cycle_ckpt_medi_matched/my_model.ckpt"
A_NAME = "axt2-???.nii"
B_NAME = "mask-???.nii"
WIDTH = 168
DATALENGTH = 1000
POOLSIZE = 50
EPOCH = 200
BATCHSIZE = 1
LOOPS = DATALENGTH // BATCHSIZE
LEARNING_RATE = 0.0002
L1_lambda = 10

def train(from_ckpt):
    tf.set_random_seed(100)
    
    true_A = tf.placeholder(tf.float32, [None,WIDTH,256,1])
    true_B = tf.placeholder(tf.float32, [None,WIDTH,256,1])
    
    fake_B = model.Gen(true_A, name="Gen_A2B")
    fake_A = model.Gen(true_B, name="Gen_B2A")
    cyc_A = model.Gen(fake_B, name="Gen_B2A")
    cyc_B = model.Gen(fake_A, name="Gen_A2B")
    
    DB_fake = model.Disc(fake_B, name="DisB")
    DA_fake = model.Disc(fake_A, name="DisA")
    DB_real = model.Disc(true_B, name="DisB")
    DA_real = model.Disc(true_A, name="DisA")

    g_loss = model.mae_criterion(DA_fake, tf.ones_like(DA_fake)) \
             + model.mae_criterion(DB_fake, tf.ones_like(DB_fake)) \
             + L1_lambda*model.abs_criterion(true_A, cyc_A) \
             + L1_lambda*model.abs_criterion(true_B, cyc_B)
    
    fake_A_sample = tf.placeholder(tf.float32,[None,WIDTH,256,1])
    fake_B_sample = tf.placeholder(tf.float32,[None,WIDTH,256,1])
    DA_fake_sample = model.Disc(fake_A_sample, name="DisA")
    DB_fake_sample = model.Disc(fake_B_sample, name="DisB")

    db_loss_real = model.mae_criterion(DB_real, tf.ones_like(DB_real))
    db_loss_fake = model.mae_criterion(DB_fake_sample, tf.zeros_like(DB_fake_sample))
    db_loss = (db_loss_real + db_loss_fake) / 2
    da_loss_real = model.mae_criterion(DA_real, tf.ones_like(DA_real))
    da_loss_fake = model.mae_criterion(DA_fake_sample, tf.zeros_like(DA_fake_sample))
    da_loss = (da_loss_real + da_loss_fake) / 2
    d_loss = da_loss + db_loss

    lr = tf.placeholder(tf.float32, name="learning_rate")
    
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
    #print(train_vars)
    
    optimizer = tf.train.AdamOptimizer(lr)
    train_gen = optimizer.minimize(g_loss, var_list=gen_vars)
    train_disc = optimizer.minimize(d_loss, var_list=disc_vars)

    horse_data = (prepare_data(DATAPATH,A_NAME) + 1.82) / 5.1 -1 
    zebra_data =  prepare_data(DATAPATH,B_NAME)  / 1.5 - 1  
    test_horse_data = horse_data[1000:,:,:,:]
    #print(test_horse_data.shape)
    test_zebra_data = zebra_data[1000:,:,:,:]
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        if from_ckpt == 0:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess,"./.ckpt-{}".format(from_ckpt))
        for epoch in range(EPOCH):
            lrate = LEARNING_RATE if epoch < 100 else LEARNING_RATE*(EPOCH - epoch)/100
            for i in range(LOOPS):
                tic = time.time()
                horse_batch = horse_data[(i*BATCHSIZE):((i+1)*BATCHSIZE),:]
                zebra_batch = zebra_data[(i*BATCHSIZE):((i+1)*BATCHSIZE),:]
                ## train generator and return fake images
                tempA, tempB, _ = sess.run([fake_A,fake_B,train_gen],feed_dict={
                    true_A:horse_batch, true_B:zebra_batch, lr:lrate})
                tempA, tempB = pool(tempA, tempB)
                ## train_discriminator using generated fake images
                sess.run(train_disc,feed_dict={
                    true_A:horse_batch, true_B:zebra_batch,
                    fake_A_sample:tempA, fake_B_sample:tempB, lr:lrate})
                print("Epoch {}: IMG {} Time: {:.4f}".format(epoch,i,time.time()-tic))
                if epoch % 10 == 9 and (i == 999):
                    saver.save(sess, CheckPointPATH, global_step = epoch)
                if epoch % 50 == 49 and (i == 999):
                    if not os.path.exists(IMGtoPATH.format(epoch)):
                        os.mkdir(IMGtoPATH.format(epoch))
                    for j in range(test_horse_data.shape[0]):
                        result_img = sess.run(fake_B,feed_dict={
                            true_A:test_horse_data[j:(j+1),:,:,:]})[0,:,:,0]
                        concat_img = np.concatenate((test_horse_data[j,:,:,0], \
                                                     result_img, \
                                                     test_zebra_data[j,:,:,0]),axis=0)
                        #print(concat_img.shape)
                        imsave(os.path.join(IMGtoPATH.format(epoch),"cycle_output{}.jpg".format(j)),concat_img)
                        
            



##def test(datapath, ckptpath, from_epch):
##    test_samples = tf.constant(prepare_data(datapath)[:10,:,:,:])
##    out_img = model.Gen(test_samples, name="Gen_A2B")
##    saver = tf.train.Saver()
##    with tf.Session() as sess:
##        saver.restore(sess, ckptpath)
##        result = sess.run(out_img)
##        for i in range(10):
##            imsave("./output{}.jpg".format(i),result[i,:,:])
##        


def prepare_data(path,name):
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
train(0)
#test("../datasets/horse2zebra/testA/","../checkpoint/horse2zebra_256/cyclegan.model-1002",1002)

