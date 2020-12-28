import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch import nn

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

def main(rank, world_size):
    # Initialisation
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    # Fix random seed
    torch.manual_seed(0)
    # Initialize network
    net = Model()
    net.cuda(rank)
    # Initialize loss function
    criterion = torch.nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.SGD(net.parameters(), 1e-4)

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    # Prepare dataset
    trainset = datasets.MNIST('./data', train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    # Prepare sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank
    )
    train_sampler = None
    # Prepare dataloader
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=False,
        num_workers=0, pin_memory=True, sampler=train_sampler)

    epoch = 0
    iteration = 0
    torch.cuda.set_device(rank)
    for _ in range(5):
        epoch += 1
       # train_loader.sampler.set_epoch(epoch)

        timestamp = time.time()
        print("Rank: {}. Before dataloader".format(rank))
        for batch in train_loader:
            print("Rank: {}. Batch loaded".format(rank))
            inputs = batch[0]
            targets = batch[1]

            iteration += 1
            inputs = inputs.cuda(rank)
            targets = targets.cuda(rank)

            output = net(inputs)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device=rank)

if __name__ == '__main__':

    # Number of GPUs to run the experiment with
    WORLD_SIZE = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    mp.spawn(main, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))