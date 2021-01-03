from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

class _DataParallel(DataParallel):
    def __init__(self, *args, **kwargs):
        super(_DataParallel,self).__init__(*args,**kwargs)

    def forward_train(self, *args, **kwargs):
        return self.module.forward_train(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def loss_cal(self, *args, **kwargs):
        return self.module.loss_cal(*args, **kwargs)

class _DistDataParallel(DDP):
    def forward_train(self, *args, **kwargs):
        return self.module.forward_train(*args, **kwargs)