import math
import time

def print_current_loss_decomp(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    print('epoch: %03d inner_iter: %5d' % (epoch, inner_iter), end=" ")
    # now = time.time()
    message = '%s niter: %07d completed: %3d%%)'%(time_since(start_time, niter_state / total_niters), niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)