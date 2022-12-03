import torch
import convnext
import torch.nn as nn
import torch.optim as optim
import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
number_of_Affine_classes = 30

#start new
#STEP3: create the model
print("****************** STEP3: create and compile the model ******************")
print("****************** STEP3.1: Create architecture  ******************")
trained_model = convnext.ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]) #tiny convnext


print("****************** STEP2.2: laod the weights  ******************")
trained_model.cuda()

print("****************** STEP2.3: copy all layers except the last  ******************")
model_for_affine = trained_model

number_features = model_for_affine.head.in_features
features = list(model_for_affine.head.children())[:-1] # Remove last layer
kwargs = {"device": "cuda"}

new_layer = nn.Linear(number_features, number_of_Affine_classes, device="cuda")
features.extend([new_layer])
model_for_affine.head = torch.nn.Sequential(*features)

checkpoint = torch.load("/home/u867803/renamed_convnext_model_best_sbatchqueued.pth")
#print(checkpoint['state_dict'])
model_for_affine.load_state_dict(checkpoint['state_dict'])
model_for_affine.cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model_for_affine.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# #end new


#define data_loader
def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    test_dir = os.path.join(root, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return test_loader, test_dataset

#load and predict
test_loader, test_dataset = data_loader("/home/u867803/Projects/Thesis/DataSets/Affine/Split_2/Split_Dataset/", 32, 64)
for i, (inputs, labels) in enumerate(test_loader):
    labels = labels.cuda(non_blocking=True)
    inputs = inputs.cuda(non_blocking=True)
    with torch.no_grad():
        # compute output
        outputs = trained_model(inputs)
        _, preds = torch.max(outputs, 1)

    #for predicted_label in enumerate(preds):
    print("image true label is: {0}", labels.data[i])
    print("model top 5 predictions are: {0}", preds)
    print()
        

