import time
import torch
import dataloader
import finalAlexNetModelpy
import torch.nn as nn
import torch.optim as optim

train_running_loss = 0.0
train_running_corrects = 0
val_running_loss = 0.0
val_running_corrects = 0  

data_root = '/home/u867803/Projects/Thesis/DataSets/Affine/Split_2/Split_Dataset'
number_of_Affine_classes = 30
batch_size = 32
workers = 64
architecture = 'alexnet_for_affine'
learning_rate = 0.01
epochs = 90
print_freq = 500
best_acc = 0.00
since = time.time()

#STEP1 & 2: load train data and validation data set
print("****************** STEP1 & 2: load train data & validation data******************")
train_loader, val_loader, train_ds, val_ds = dataloader.data_loader(data_root, batch_size, workers)
print("train dataset size: {0}", len(train_ds))
print("validation dataset size: {0}", len(val_ds))

#STEP3: create the model
print("****************** STEP3: create and compile the model ******************")
print("****************** STEP3.1: Create architecture  ******************")
trained_model = finalAlexNetModelpy.AlexNet()


print("****************** STEP2.2: laod the weights  ******************")
checkpoint = torch.load("/home/u867803/Projects/Thesis/Scripts/AlexNet/1/alex_model_best_sbatchqueued.pth")
trained_model.load_state_dict(checkpoint['state_dict'])
trained_model.cuda()
print(trained_model)

print("****************** STEP2.3: copy all layers except the last  ******************")
model_for_affine = trained_model

# freeze the trained layers
for param in model_for_affine.parameters():
   param.requires_grad = False

# Modify the last layer
number_features = model_for_affine.classifier[6].in_features
features = list(model_for_affine.classifier.children())[:-1] # Remove last layer
features.extend([torch.nn.Linear(number_features, number_of_Affine_classes, device="cuda")])
model_for_affine.classifier = torch.nn.Sequential(*features)

model_for_affine.cuda
model_for_affine.cuda
print(model_for_affine)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model_for_affine.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

# #STEP3: Train and validation
print("****************** STEP3: Train and validation ******************")

for epoch in range(0, epochs):
    # print('Epoch {}/{}'.format(epoch+1, epochs))
    # print('-' * 10)
    finalAlexNetModelpy.adjust_learning_rate(optimizer, epoch, learning_rate)
    #print("here {0}", 1)
    #**************** train for one epoch************************#
    train_running_corrects, train_running_loss = finalAlexNetModelpy.train_epoch(train_loader, model_for_affine, criterion, optimizer, epoch, print_freq)
    train_epoch_loss = train_running_loss /  len(train_ds)
    train_epoch_acc = train_running_corrects.double() / len(train_ds)
    
    #**************** validate for one epoch************************#

    # evaluate on validation set
    val_running_corrects, val_running_loss = finalAlexNetModelpy.validate_epoch(val_loader, model_for_affine, criterion, print_freq)

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

    finalAlexNetModelpy.save_checkpoint({
        'epoch': epoch + 1,
        'arch': architecture,
        'state_dict': model_for_affine.state_dict(),
        'optimizer': optimizer.state_dict()
    }, is_best, architecture + '_sbatchQueued.pth')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))