import os
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import shutil
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
import sys

liveloss = PlotLosses()
train_running_loss = 0.0
train_running_corrects = 0
val_running_loss = 0.0
val_running_corrects = 0  

def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, train_dataset, val_dataset

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='alex_checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'alex_model_best_sbatchqueued.pth')

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # shape is 55 x 55 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 27 x 27 x 64

            nn.Conv2d(64, 192, kernel_size=5, padding=2), # shape is 27 x 27 x 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 13 x 13 x 192

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # shape is 6 x 6 x 256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def train_epoch(train_loader, model, criterion, optimizer, epoch, print_freq):
   
    local_train_running_loss= 0.0
    local_train_running_corrects= 0

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        labels = labels.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
        
        # statistics
        local_train_running_loss += loss.item() * inputs.size(0)
        local_train_running_corrects += torch.sum(preds == labels.data)
        #print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(train_loader), loss.item() * inputs.size(0)), end="")
        #sys.stdout.flush()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return local_train_running_corrects, local_train_running_loss
                
def validate_epoch(val_loader, model, criterion, print_freq):

    local_val_running_loss = 0.0
    local_val_running_corrects = 0

    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, labels) in enumerate(val_loader):
        labels = labels.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)
        
        # zero the parameter gradients
        #optimizer.zero_grad()

        with torch.no_grad():
            # compute output
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        # statistics
        local_val_running_loss += loss.item() * inputs.size(0)
        local_val_running_corrects += torch.sum(preds == labels.data)
        
        #print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(val_loader), loss.item() * inputs.size(0)), end="")
        #sys.stdout.flush()
    return local_val_running_corrects, local_val_running_loss

data_root = '/home/u867803/Projects/Thesis/DataSets/ImageNet'
batch_size = 256
workers = 64
architecture = 'alexnet'
learning_rate = 0.01
epochs = 90
print_freq = 500
best_acc = 0.00
model = AlexNet()
since = time.time()
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
train_loader, val_loader, train_ds, val_ds = data_loader(data_root, batch_size, workers)

for epoch in range(0, epochs):
    # print('Epoch {}/{}'.format(epoch+1, epochs))
    # print('-' * 10)
    adjust_learning_rate(optimizer, epoch, learning_rate)
    
    #**************** train for one epoch************************#
    train_running_corrects, train_running_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, print_freq)
    train_epoch_loss = train_running_loss /  len(train_ds)
    train_epoch_acc = train_running_corrects.double() / len(train_ds)
    
    #**************** validate for one epoch************************#

    # evaluate on validation set
    val_running_corrects, val_running_loss = validate_epoch(val_loader, model, criterion, print_freq)

    val_epoch_loss = val_running_loss /  len(val_ds)
    val_epoch_acc = val_running_corrects.double() / len(val_ds)
    
    #**************** update the liveloss for one epoch************************#
    something = {
            'log loss': train_epoch_loss,
            'val_log loss': val_epoch_loss,
            'accuracy': train_epoch_acc,
            'val_accuracy': val_epoch_acc
        }
    #liveloss.update(something.cpu())
                
    #liveloss.draw()
    # print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, train_epoch_acc))
    # print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))
    # print()
    print('{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(train_epoch_loss, train_epoch_acc,val_epoch_loss, val_epoch_acc))
    print()

    is_best = False
    
    # remember the best prec@1 and save checkpoint
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        best = epoch + 1
        is_best = True

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': architecture,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, is_best, architecture + '_sbatchQueued.pth')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))