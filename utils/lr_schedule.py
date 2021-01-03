
def adjust_learning_rate(optimizer, gamma, step, lr):

    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, step_now, warmup_step, lr):
    lr = lr / warmup_step * step_now
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr