import torch
from options.train_options  import TrainOptions
from os.path import join as pjoin
import os
from utils.paramUtils import t2m_kinematic_chain
import numpy as np
from utils.word_vectorizer import WordVectorizer
from torch.utils.data import DataLoader
from data_utils.dataset import MotionDatasetV2

if __name__ == "__main__":
    print('Is cuda device: ', torch.cuda.is_available())
    parser = TrainOptions()
    options = parser.parse()
    options.device = torch.device("cpu" if options.gpu_id==-1 else "cuda:" + str(options.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    if options.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(options.gpu_id)
    
    options.save_root = pjoin(options.checkpoints_dir, 'HumanML3D', options.name)
    options.model_dir = pjoin(options.checkpoints_dir, 'model')
    options.meta_dir = pjoin(options.save_root, 'meta')
    options.eval_dir = pjoin(options.save_root, 'animation')
    options.log_dir = pjoin('./log', options.dataset_name, options.name)

    os.makedirs(options.model_dir, exist_ok=True)
    os.makedirs(options.meta_dir, exist_ok=True)
    os.makedirs(options.eval_dir, exist_ok=True)
    os.makedirs(options.log_dir, exist_ok=True)

    options.data_root = './data/HumanML3D/HumanML3D'
    options.motion_dir = pjoin(options.data_root, 'new_joint_vecs')
    options.text_dir = pjoin(options.data_root, 'texts')
    options.joints_num = 22
    options.max_motion_length = 196
    dim_pose = 263
    radius = 4
    fps = 20
    kinematic_chain = t2m_kinematic_chain


    mean = np.load(pjoin(options.data_root, 'Mean.npy'))
    std = np.load(pjoin(options.data_root, 'Std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(options.data_root, 'train.txt')
    val_split_file = pjoin(options.data_root, 'val.txt')

    train_dataset = MotionDatasetV2(options, mean, std, train_split_file)
    val_dataset = MotionDatasetV2(options, mean, std, val_split_file)
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, drop_last=True, num_workers=1,
                              shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size, drop_last=True, num_workers=1,
                            shuffle=False, pin_memory=True)

