import torch
import torch.nn.functional as F

class Logger(object):
  def __init__(self, log_dir):
    # self.writer = tf.summary.create_file_writer(log_dir)
    pass

  def scalar_summary(self, tag, value, step):
    #   with self.writer.as_default():
    #       tf.summary.scalar(tag, value, step=step)
    #       self.writer.flush()
    pass

class MotionTrainer(object):

    def __init__(self, args, movement_enc, movement_dec):
        self.opt = args
        self.movement_enc = movement_enc
        self.movement_dec = movement_dec
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.sml1_criterion = torch.nn.SmoothL1Loss()
            self.l1_criterion = torch.nn.L1Loss()
            self.mse_criterion = torch.nn.MSELoss()