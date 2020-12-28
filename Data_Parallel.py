from torch.nn.parallel import DataParallel

class _DataParallel(DataParallel):
    def __init__(self, *args, **kwargs):
        super(_DataParallel,self).__init__(*args,**kwargs)

    def forward_train(self, *args, **kargs):
        return self.module.forward_train(*args, **kargs)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)