import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_bird_data():
    transform_train = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomCrop(size=224, padding=8, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = torchvision.datasets.ImageFolder(root='birds/train', transform=transform_train)
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size-train_size

    trainset, valset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='birds/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    
    classes = open("birds/names.txt").read().strip().split("\n")
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k,v in idx_to_class.items()}
    return {'train': trainloader, 'test': testloader, 'val':valloader, 'to_class': idx_to_class, 'to_name':idx_to_name}


def train(net, dataloader,valloader, epochs=1, start_epoch=0, max_lr = 1.0, base_lr=0.00001, verbose=1,momentum=0.9, decay=0.0005, print_every=10):
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.0001,weight_decay=0.001)
    #optimizer = optim.SGD(net.parameters(),lr=lr,momentum=momentum,weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=600, step_size_down=None, mode='triangular', gamma=1.0,cycle_momentum=False)

    for epoch in range(start_epoch, epochs):
        loss_so_far = 0.0

        for i, batch in enumerate(dataloader, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_so_far += loss.item()

            if i % print_every == print_every-1:    # print every 10 mini-batches
                if verbose:
                  print('[%d, %5d] loss: %.3f' % (epoch, i + 1, loss_so_far / print_every))
                loss_so_far = 0.0
        
        
        validate(net,valloader)

        scheduler.step()

def validate(net, dataloader):
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    corr = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            corr += (predicted == labels).sum().item()

    acc = 100 * corr/total
    avg_loss = total_loss/len(dataloader)
    print('Val Loss:{:.3f}, Acc:{:.3f}%'.format(avg_loss, acc))

def predict(net, dataloader, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    net.to(device)
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            if i%100 == 0:
                print(i)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            fname, _ = dataloader.dataset.samples[i]
            out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
    out.close()

if __name__ == '__main__':
    data = get_bird_data()
    dataiter = iter(data['train'])

    number_of_birds = 555

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    print(num_features)
    # model.fc = nn.Sequential(
    #                   nn.Linear(num_features, 256), 
    #                   nn.ReLU(),
    #                   nn.Dropout(0.5),
    #                   nn.Linear(256, number_of_birds),
    #                   nn.LogSoftmax(dim=1)
    # )

    # model.fc = nn.Sequential(
    #                         nn.Linear(num_features,number_of_birds),
    #                         nn.LogSoftmax(dim=1)
    #             )

    # model.fc = nn.Linear(num_features,number_of_birds)

    # model.fc = nn.Sequential(
    #                   nn.Linear(num_features, 256), 
    #                   nn.ReLU(),
    #                   nn.Dropout(0.5),
    #                   nn.Linear(256, number_of_birds),
    # )

    model.fc = nn.Sequential(
                      nn.Linear(num_features, 256), 
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(256, number_of_birds)
    )

    train(model, data['train'],data['val'], epochs=20, max_lr=0.1, print_every=10)
    predict(model, data['test'],"preds.csv")