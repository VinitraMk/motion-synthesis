from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class TrainOptions():
    def __init__(self):
        self.parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        self.initialized = False
    
    def initialize(self):
        every_e_interval = 2
        self.parser.add_argument('--name', type = str, default = 'test', help = 'Name of the trial')
        self.parser.add_argument('--checkpoints_dir', type = str, default = './checkpoints', help = 'Models and other data are saved here')
        self.parser.add_argument('--experiment_dir', type = str, default = './exp_results/vq-vae-setup', help = 'Save experiment loss logs and reports')
        self.parser.add_argument('--dataset_name', type = str, default = 'HumanML3D')
        self.parser.add_argument('--gpu_id', type = int, default = -1, help = 'GPU id')
        self.parser.add_argument('--window_size', type=int, default=40, help="Length of motion clips for reconstruction")
        self.parser.add_argument('--dim_txt_hidden', type = int, default = 512, help = 'Dimension of hidden layer in text encoder')
        self.parser.add_argument('--feat_bias', type = int, default = 5, help = 'Layers of GRU')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
        self.parser.add_argument('--kl_warmup_epoch', type=int, default=30, help='Number of warmup epochs')
        self.parser.add_argument('--kl_beta_max', type=float, default=0.1, help='Maximum value of KL beta')
        self.parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for improvement')
        self.parser.add_argument('--min_loss_delta', type=float, default=0.01, help='Delta difference for loss improvement')
        self.parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
        self.parser.add_argument('--log_every', type=int, default=every_e_interval, help='Frequency of printing training progress (by iteration)')
        self.parser.add_argument('--save_every_e', type=int, default=2, help='Frequency of saving models (by epoch)')
        self.parser.add_argument('--eval_every_e', type=int, default=every_e_interval, help='Frequency of animation results (by epoch)')
        self.parser.add_argument('--is_continue', action="store_true", help='Is this trail continued from previous trail?')
        self.parser.add_argument('--max_epoch', type=int, default=2000, help='Training iterations')
        self.parser.add_argument('--save_latest', type=int, default=every_e_interval, help='Frequency of saving models (by iteration)')
        self.parser.add_argument('--dataset_mode', type=str, default="micro", help='Frequency of saving models (by iteration)')
        self.parser.add_argument('--stage', type=str, default = 'generation', help = 'Stage of the pipeline (autoencoder or generation)')
        self.initialized = True

    def parse(self, args = []):
        if not(self.initialized):
            self.initialize()
        
        self.options, _ = self.parser.parse_known_args(args = args)
        self.options.is_train = True

        return self.options

