
# coding: utf-8

# In[ ]:

import os
import sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2' # set a GPU (with GPU Number)
home = os.path.expanduser("~")
sys.path.append(home + '/tflib/')        # path for 'tflib' folder
from SRNet import *


# In[ ]:

train_batch_size = 1
valid_batch_size = 10
max_iter = 10000
train_interval=100
#valid_interval=5000
valid_interval=500
save_interval=500
num_runner_threads=10

# Cover and Stego directories for training and validation. For the spatial domain put cover and stego images in their 
# corresponding direcotries. For the JPEG domain, decompress images to the spatial domain without rounding to integers and 
# save them as '.mat' files with variable name "im". Put the '.mat' files in thier corresponding directoroies. Make sure 
# all mat files in the directories can be loaded in Python without any errors.

TRAIN_COVER_DIR = '/data/wangke/ImageSteg/photo/unsplash/steg/1024/401/center/SRNet/media/TRN/Cover_1000/'
TRAIN_STEGO_DIR = '/data/wangke/ImageSteg/photo/unsplash/steg/1024/401/center/SRNet/media/TRN/JUNI_1000/'

VALID_COVER_DIR = '/data/wangke/ImageSteg/photo/unsplash/steg/1024/401/center/SRNet/media/VAL/Cover_100/'
VALID_STEGO_DIR = '/data/wangke/ImageSteg/photo/unsplash/steg/1024/401/center/SRNet/media/VAL/JUNI_100/'
    
#train_gen = partial(gen_flip_and_rot,                     TRAIN_COVER_DIR, TRAIN_STEGO_DIR ) 
#valid_gen = partial(gen_valid,                     VALID_COVER_DIR, VALID_STEGO_DIR)

LOG_DIR = '/data/wangke/ImageSteg/photo/unsplash/steg/1024/401/center/SRNet/media/LogFiles/JUNI_75_04/'  # path for a log direcotry 
# load_path = LOG_DIR + 'Model_460000.ckpt'  # continue training from a specific checkpoint
load_path=None                              # training from scratch

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

train_ds_size = len(glob(TRAIN_COVER_DIR + '/*')) * 2
valid_ds_size = len(glob(VALID_COVER_DIR +'/*')) * 2
print ('train_ds_size: %i'%train_ds_size)
print ('valid_ds_size: %i'%valid_ds_size)

if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation")
    
#optimizer = AdamaxOptimizer
#boundaries = [400000]     # learning rate adjustment at iteration 400K
#values = [0.001, 0.0001]  # learning rates

#train(SRNet, train_gen, valid_gen , train_batch_size, valid_batch_size, valid_ds_size,       optimizer, boundaries, values, train_interval, valid_interval, max_iter,      save_interval, LOG_DIR,num_runner_threads, load_path)


# In[ ]:

# Testing 
# Cover and Stego directories for testing
TEST_COVER_DIR = '/data/wangke/ImageSteg/photo/unsplash/steg/1024/401/center/SRNet/media/TST/Cover/'
TEST_STEGO_DIR = '/data/wangke/ImageSteg/photo/unsplash/steg/1024/401/center/SRNet/media/TST/JUNI_75_04/'

test_batch_size=10
LOG_DIR = '/data/wangke/ImageSteg/photo/unsplash/steg/1024/401/center/SRNet/media/LogFiles/JUNI_75_04/' 
LOAD_CKPT = LOG_DIR + 'Model_100000.ckpt'        # loading from a specific checkpoint

test_gen = partial(gen_valid,                     TEST_COVER_DIR, TEST_STEGO_DIR)

test_ds_size = len(glob(TEST_COVER_DIR + '/*')) * 2
print ('test_ds_size: %i'%test_ds_size)

if test_ds_size % test_batch_size != 0:
    raise ValueError("change batch size for testing!")

test_dataset(SRNet, test_gen, test_batch_size, test_ds_size, LOAD_CKPT)


# In[ ]:



