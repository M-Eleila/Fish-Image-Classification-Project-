import time
import torch
import dataloader
import convnext
import torch.nn as nn
import torch.optim as optim
import ModelExecutionHelper

unfreeze_trained_layers = True
train_running_loss = 0.0
train_running_corrects = 0
val_running_loss = 0.0
val_running_corrects = 0  

data_root = '/home/u867803/Projects/Thesis/DataSets/Robo/Data_set_robo2'
number_of_Robo_classes = 26
batch_size = 32
workers = 64
architecture = 'convnext_for_robo'
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
trained_model = convnext.ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]) #tiny convnext
number_features = trained_model.head.in_features
features = list(trained_model.head.children())[:-1] # Remove last layer
kwargs = {"device": "cuda"}

new_layer = nn.Linear(number_features, 30, device="cuda")
features.extend([new_layer])
trained_model.head = torch.nn.Sequential(*features)

print("****************** STEP2.2: laod the weights  ******************")
checkpoint = torch.load("/home/u867803/renamed_convnext_model_best_sbatchqueued.pth")
trained_model.load_state_dict(checkpoint['state_dict'])

trained_model.cuda()
print(trained_model)

print("****************** STEP2.3: copy all layers except the last  ******************")
model_for_robo = trained_model

# Disable gradients on all model parameters to freeze the weights
for param in model_for_robo.parameters():
    param.requires_grad = unfreeze_trained_layers

number_features = model_for_robo.head[0].in_features
print(number_features)
features = list(model_for_robo.head[0].children())[:-1] # Remove last layer
kwargs = {"device": "cuda"}

new_layer = nn.Linear(number_features, number_of_Robo_classes, device="cuda")
#features.extend([torch.nn.Linear(number_features, number_of_Robo_classes, **kwargs)])
features.extend([new_layer])
model_for_robo.head = torch.nn.Sequential(*features)

#model_for_robo.head = nn.Sequential(nn.Linear(768, number_of_Robo_classes, device="cuda"))

for param in model_for_robo.head.parameters():
    param.requires_grad = unfreeze_trained_layers

# Unfreeze the last stage
for param in model_for_robo.stages[3].parameters():
    param.requires_grad = unfreeze_trained_layers

model_for_robo.cuda
print(model_for_robo)

# criterion = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(model_for_robo.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

# # #STEP3: Train and validation
# print("****************** STEP3: Train and validation ******************")

# for epoch in range(0, epochs):
#     # print('Epoch {}/{}'.format(epoch+1, epochs))
#     # print('-' * 10)
#     ModelExecutionHelper.adjust_learning_rate(optimizer, epoch, learning_rate)
#     #print("here {0}", 1)
#     #**************** train for one epoch************************#
#     train_running_corrects, train_running_loss = ModelExecutionHelper.train_epoch(train_loader, model_for_robo, criterion, optimizer, epoch, print_freq)
#     train_epoch_loss = train_running_loss /  len(train_ds)
#     train_epoch_acc = train_running_corrects.double() / len(train_ds)
    
#     #**************** validate for one epoch************************#

#     # evaluate on validation set
#     val_running_corrects, val_running_loss = ModelExecutionHelper.validate_epoch(val_loader, model_for_robo, criterion, print_freq)

#     val_epoch_loss = val_running_loss /  len(val_ds)
#     val_epoch_acc = val_running_corrects.double() / len(val_ds)
    
#     #**************** update the liveloss for one epoch************************#
#     something = {
#             'log loss': train_epoch_loss,
#             'val_log loss': val_epoch_loss,
#             'accuracy': train_epoch_acc,
#             'val_accuracy': val_epoch_acc
#         }
#     #liveloss.update(something.cpu())
                
#     #liveloss.draw()
#     # print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, train_epoch_acc))
#     # print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))
#     # print()
#     print('{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(train_epoch_loss, train_epoch_acc,val_epoch_loss, val_epoch_acc))
#     print()

#     is_best = False
    
#     # remember the best prec@1 and save checkpoint
#     if val_epoch_acc > best_acc:
#         best_acc = val_epoch_acc
#         best = epoch + 1
#         is_best = True

#     ModelExecutionHelper.save_checkpoint({
#         'epoch': epoch + 1,
#         'arch': architecture,
#         'state_dict': model_for_robo.state_dict(),
#         'optimizer': optimizer.state_dict()
#     }, is_best, architecture + '_sbatchQueued.pth')

# time_elapsed = time.time() - since
# print('Training complete in {:.0f}m {:.0f}s'.format(
#     time_elapsed // 60, time_elapsed % 60))
# print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))