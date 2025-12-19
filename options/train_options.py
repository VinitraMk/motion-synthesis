from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class TrainOptions():
    def __init__(self):
        self.parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        self.initialized = False
    
    def initialize(self):
        self.parser.add_argument('--name', type = str, default = 'test', help = 'Name of the trial')
        self.parser.add_argument('--checkpoints_dir', type = str, default = './checkpoints', help = 'Models and other data are saved here')
        self.parser.add_argument('--dataset_name', type = str, default = 'HumanML3D')
        self.parser.add_argument('--gpu_id', type = int, default = -1, help = 'GPU id')
        self.parser.add_argument('--window_size', type=int, default=40, help="Length of motion clips for reconstruction")
        self.parser.add_argument('--dim_txt_hidden', type = int, default = 512, help = 'Dimension of hidden layer in text encoder')
        self.parser.add_argument('--dim_att_vec', type = int, default = 512, help = 'Dimension of attention vector')
        self.parser.add_argument('--dim_z', type = int, default = 128, help = 'Dimension of latent Gaussian vector')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
        self.parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

        self.initialized = True

    def parse(self):
        if not(self.initialized):
            self.initialize()
        
        self.options = self.parser.parse_args()
        self.options.is_train = True

        return self.options

