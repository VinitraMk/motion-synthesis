import torch
from options.train_options  import TrainOptions
from os.path import join as pjoin
import os
from utils.paramUtils import t2m_kinematic_chain
import numpy as np
from utils.word_vectorizer import WordVectorizer
from torch.utils.data import DataLoader
from data_utils.dataset import MotionDatasetV2
from data_utils.dataset import PartMotionDatasetV2
from networks.nn import MotionVQVAE
from networks.trainers import MotionVQVAETrainer
from torch.utils.data import Subset
from networks.nn_validator import VQVAEValidator
import json

if __name__ == "__main__":
    parser = TrainOptions()
    options = parser.parse(args = ['--max_epoch', '600'])
    options.gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    options.device = torch.device("cpu" if options.gpu_id==-1 else "cuda:" + str(options.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    if options.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(options.gpu_id)

    print('\nDevice used: ', options.device)
    options.save_root = pjoin(options.checkpoints_dir, 'HumanML3D', options.name)
    options.model_dir = pjoin(options.checkpoints_dir, 'model')
    options.meta_dir = pjoin(options.save_root, 'meta')
    options.eval_dir = pjoin(options.save_root, 'animation')
    options.log_dir = pjoin('./log', options.dataset_name, options.name)
    options.experiment_dir = './exp_results/vq-vae-setup/beta_0.1_full'
    options.output_dir = options.experiment_dir
    options.save_every_e = 10
    options.is_train = False
    options.is_continue = False
    options.dataset_mode = "debug"
    options.batch_size = 64
    options.model_filename = 'vqvae_beta_0.1_full_v0.tar'

    os.makedirs(options.model_dir, exist_ok=True)
    os.makedirs(options.meta_dir, exist_ok=True)
    os.makedirs(options.eval_dir, exist_ok=True)
    os.makedirs(options.log_dir, exist_ok=True)

    options.data_root = './data/HumanML3D'
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
    train_split_fn = 'train.txt'
    val_split_fn = 'val.txt'

    if options.dataset_mode == "debug":
        train_split_fn = 'train_debug.txt'
        val_split_fn = 'val_debug.txt'
    elif options.dataset_mode == "micro":
        train_split_fn = 'train_micro.txt'
        val_split_fn = 'val_micro.txt'

    train_split_file = pjoin(options.data_root, train_split_fn)
    val_split_file = pjoin(options.data_root, val_split_fn)

    if options.dataset_mode == "micro":
        subset_path = os.path.join(options.meta_dir, "micro_subsets.json")
        par_train_dataset = PartMotionDatasetV2(options, mean, std, train_split_file)
        par_val_dataset = PartMotionDatasetV2(options, mean, std, val_split_file)

        if os.path.exists(subset_path):
            with open(subset_path, "r") as f:
                subsets = json.load(f)
            micro_train_indices = subsets["micro_train_indices"]
            micro_val_indices = subsets["micro_val_indices"]
        else:
            seed = 42
            rng = np.random.default_rng(seed)

            all_train_indices = np.arange(len(par_train_dataset))
            all_val_indices = np.arange(len(par_val_dataset))

            micro_train_indices = rng.permutation(all_train_indices)[:80].tolist()
            micro_val_indices = rng.permutation(all_val_indices)[:30].tolist()

            with open(subset_path, "w") as f:
                json.dump({
                    "seed": seed,
                    "micro_train_indices": micro_train_indices,
                    "micro_val_indices": micro_val_indices,
                }, f, indent=4)
        train_dataset = Subset(par_train_dataset, micro_train_indices)
        val_dataset = Subset(par_val_dataset, micro_val_indices)
    else:
        train_dataset = PartMotionDatasetV2(options, mean, std, train_split_file)
        val_dataset = PartMotionDatasetV2(options, mean, std, val_split_file)

    print('\nTotal number of snippets in train: ', len(train_dataset))
    print('Total number of snippets in val: ', len(val_dataset))
    sample_motion = train_dataset[8]
    print('Sample data shape: ', sample_motion['motion_parts'].shape, sample_motion['text'])
    Dp_max = sample_motion['motion_parts'].shape[-1]

    
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, drop_last=True, num_workers=1,
                              shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size, drop_last=True, num_workers=1,
                            shuffle=False, pin_memory=True)
    
    vqvae = MotionVQVAE(
        input_dim=Dp_max,
        enc_hidden_dim=1024,
        dec_hidden_dim=1024,
        latent_dim=256,
        num_embeddings=512,
        beta=0.1
    )
        
    if options.is_train: 
        trainer = MotionVQVAETrainer(options, vqvae = vqvae)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader)

    test_model_filepath = pjoin(options.model_dir, options.model_filename)

    if not(options.is_train) and os.path.exists(test_model_filepath):
        vqvae_model_dict = torch.load(pjoin(options.model_dir, 'vqvae_beta_0.2.tar'), map_location = options.device)

        vqvae.load_state_dict(vqvae_model_dict['vqvae'])

        vqvae_validator = VQVAEValidator(
            opt = options,
            vqvae=vqvae,
            train_dataloader=train_loader,
            val_dataloader = val_loader
        )
        vqvae_validator.validate()
    else:
        print("Invalid mode or model file doesn't exist!")
