import AverageMeter
import time
import torch.utils.data
import shutil

def train_epoch(train_loader, model, criterion, optimizer, epoch, print_freq):
   
    local_train_running_loss= 0.0
    local_train_running_corrects= 0

    batch_time = AverageMeter.AverageMeter()
    data_time = AverageMeter.AverageMeter()

    #print("here {0}", 2)
    # switch to train mode
    model.train()
    #print("here {0}", 3)

    end = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #print("here {0}", 4)

        labels = labels.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        optimizer.zero_grad()
        # forward
        # track history if only in train
        #print("inputs is cuda {0}", inputs.is_cuda)
        
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
        #print("here {0}", 5)
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

    batch_time = AverageMeter.AverageMeter()

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

def save_checkpoint(state, is_best, filename='alex_checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'alex_model_best_sbatchqueued.pth')

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr