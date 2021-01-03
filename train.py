import time
import argparse
import logging
import torch
import pickle
from tqdm import tqdm
from torch.backends import cudnn
from utils.logger import setup_logger
import torch.distributed as dist
import torch.utils.data as Data
from dataset.voc_detector_collate import voc_detector_co
from model.detector.ssd_det import SSD_DET
from dataset.VOC import VOC, Compose
from dataset.transform import *
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from utils.checkpoint import load_model_optimizer_checkpoint, save_checkpoint
from utils.Data_Parallel import _DistDataParallel, _DataParallel
from utils.lr_schedule import warmup_learning_rate, adjust_learning_rate
from utils.current_time import get_current_time, time_to_hour
from eval import VOCMap

def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix

def test_(model, dataloader, epoch, local_rank):
    results = []
    img_infos_list = []
    for img, targets, img_infos in tqdm(dataloader):
        with torch.no_grad():
            img = img.to(local_rank)
            bbox_results = model.module.simple_test(img, img_infos, rescale=True)
            results.append(bbox_results)
            img_infos_list.append(img_infos)
    data = {'result': results, 'img_info': img_infos_list}
    with open('data_{}.pkl'.format(epoch), 'wb') as p:
        pickle.dump(data, p)
    results = []
    for i in data['result']:
        for j in i:
            results.append(j)
    img_list = []
    for i in data['img_info']:
        for k in i:
            img_list.append(k['id'])
    all_boxes = transpose(results)
    return all_boxes, img_list


def train(arg, local_rank):
    logger = logging.getLogger(name='ssd.train')
    logger.info('training.')
    batch_size=32
    checkpoint_from = '/workspace/code/ssd-pytorch/trainvalaug_checkpoint240.pth'

    #dataset
    dataset_root = '/workspace/data/VOC2007'
    dataset = VOC(dataset_root, use_set='trainval',
                  transforms=Compose([PhotometricDistort(delta=32,
                                                         contrast_range=(0.5, 1.5),
                                                         saturation_range=(0.5, 1.5),
                                                         hue_delta=18),
                                      expand(mean=(123.675, 116.28, 103.53),
                                             to_rgb=True,
                                             expand_ratio=(1, 4)),
                                      MinIoURandomCrop(min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                                                       min_crop_size=0.3,
                                                       bbox_clip_border=True),
                                      resize(resize_size=(300, 300),
                                             keep_ratio=False),
                                      normalize(mean=[123.675, 116.28, 103.53],
                                                std=[1, 1, 1],
                                                to_rgb=True),
                                      flip_random(flip_ratio=0.5, direction='horizontal'),
                                      DefualtFormat(), ]))
    train_sample = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size,
                                 num_workers=0,
                                 collate_fn=voc_detector_co,
                                 pin_memory=False, sampler=train_sample
                                 )

    test_dataset = VOC(dataset_root, use_set='test', transforms=Compose(
        [resize(), normalize(), DefualtFormat()]))
    #test_sample = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=0,
                                 collate_fn=voc_detector_co,
                                 pin_memory=False, shuffle=True
                                      #sampler=test_sample
                                 )

    #model
    model_classes = 20
    resume = 0
    pretrained = 'vgg16_caffe-292e1171.pth'
    model = SSD_DET(model_classes)

    if resume:
        start_epoch = load_model_optimizer_checkpoint(model, None, checkpoint=checkpoint_from)
    else:
        model.init_weights(pretrained=pretrained)
        start_epoch = 0
    model = model.to(local_rank)
    model = _DistDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.1)
    if resume:
        load_model_optimizer_checkpoint(None, optimizer,checkpoint=checkpoint_from)

    #training
    start_time = time.time()
    start_step = start_epoch * len(dataloader)

    save_during_epoch = 40
    epoch_steps = [160,220]
    eval_period = 40
    total_epoch = 280
    warmup_step = 200
    checkpoint_prefix = 'workdirs/ck'

    total_loss = []
    cla_loss = []
    loc_loss = []

    for epoch in range(start_epoch, total_epoch+1):
        model.train()
        dataloader.sampler.set_epoch(epoch)
        print_epoch = epoch + 1
        #if dist.get_rank()==0:
        logger.info("Epoch {}/{}".format(epoch+1, total_epoch))
        logger.info('-' * 10)

        model.train(True)
        epoch_loss = 0
        epoch_loc = 0
        epoch_cla = 0
        for i, (images, targets, img_infos) in enumerate(dataloader):
            print_step = i + 1
            step_now = epoch * len(dataloader) + print_step
            if step_now <= warmup_step:
                warmup_learning_rate(optimizer, step_now, warmup_step, lr=arg.lr)
            images = images.to(local_rank)

            with torch.no_grad():
                gt_bboxes = [gt[0].to(local_rank) for gt in targets]
                gt_labels = [gt[1].to(local_rank) for gt in targets]

            optimizer.zero_grad()

            loss_cla = 0
            loss_loc = 0

            loss_c, loss_l = model.forward_train(images, gt_bboxes, gt_labels)
            for k in loss_c:
                loss_cla += k
            for j in loss_l:
                loss_loc += j
            loss = loss_cla + loss_loc
            loss.backward()
            optimizer.step()

            # count avg loss
            if dist.is_available() and dist.is_initialized():
                loss_value = loss.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
                loss_cla_value = loss_cla.data.clone()
                dist.all_reduce(loss_cla_value.div_(dist.get_world_size()))
                loss_loc_value = loss_loc.data.clone()
                dist.all_reduce(loss_loc_value.div_(dist.get_world_size()))

                cla_loss.append(loss_cla_value)
                loc_loss.append(loss_loc_value)
                total_loss.append(loss_value)

                with torch.no_grad():
                    epoch_loss += loss_value
                    epoch_loc += loss_loc_value
                    epoch_cla += loss_cla_value

            if print_step % 10 == 0:
                avg_step = (time.time() - start_time) / (step_now - start_step)
                rest_step = (total_epoch - epoch) * len(dataloader) - print_step
                rest_time = avg_step * rest_step
                hour, minutes, sec = time_to_hour(rest_time)
                logger.info(
                    'epoch {}, iter {}/{}, loss_cla: {:.3f}, loss_loc: {:.3f}, '
                    'loss: {:.3f}, lr: {:.3e}, etc:{}h{}m{}s'.format(
                        print_epoch, print_step, len(dataloader), loss_cla_value.item(), loss_loc_value.item(), loss_value.item(),
                        optimizer.param_groups[0]['lr'],hour,minutes,sec))
                start_time = time.time()
                start_step = step_now

            if print_epoch in epoch_steps:
                scheduler.step()
                logger.info('sum_iter {}'.format(epoch_steps))

        logger.info('epoch {} loss_cla: {:.4f}, loss_loc: {:.4f} loss: {:.4f}'.format
                (print_epoch, epoch_cla.item()/(len(dataloader)),
                 epoch_loc.item()/(len(dataloader)), epoch_loss.item()/(len(dataloader))))

        if dist.get_rank() == 0 and print_epoch % save_during_epoch == 0 :#and epoch != 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_prefix)
        dist.barrier()

        if print_epoch % eval_period == 0:
            logger.info('start eval')
            model.eval()
            if dist.get_rank()==0:
                all_boxes, img_list = test_(model,test_dataloader,epoch,local_rank)
                ap = VOCMap('/workspace/data/', 21, num_images=len(test_dataset))
                ap(all_boxes, img_list)
            dist.barrier()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--lr', help='learning rate', type=float, default=2e-4)
    parser.add_argument('--local_rank', help='learning rate', type=int, default=-1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    #dist training setting
    dist.init_process_group(backend='gloo')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    output_dir = 'workdirs/'
    logger = setup_logger(name="ssd.train", output=output_dir,distributed_rank=local_rank)
    logger.info(args)
    cudnn.benchmark = True
    train(args, local_rank)

if __name__ == '__main__':
    main()



