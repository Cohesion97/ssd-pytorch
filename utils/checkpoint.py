import torch
from model.load_pretrained import weights_to_cpu, load_checkpoint

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {'state_dict':weights_to_cpu(model.state_dict()),
                  'optimizer':optimizer.state_dict(),
                  'epoch':epoch}
    filename = filename.split('.')[0] + str(epoch) + '.pth'
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()

    print('save model{} checkpoint to {}'.format(epoch,filename))

def load_model_optimizer_checkpoint(model, optimizer, checkpoint, map_location='default', strict=False):
    if model is not None:
       print('load model checkpoint from {}'.format(checkpoint))
       if map_location == 'default':

           if torch.cuda.is_available():
               device_id = torch.cuda.current_device()
               checkpoint = load_checkpoint(model,
                   checkpoint,
                   map_location=lambda storage, loc: storage.cuda(device_id))
           else:
               checkpoint = load_checkpoint(model, checkpoint)
       else:
           checkpoint = load_checkpoint(model, checkpoint, map_location=map_location)
    if 'optimizer' in checkpoint and optimizer is not None:
       optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'] if 'epoch' in checkpoint else 0