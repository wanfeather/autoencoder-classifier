import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from unet import Classifier
from dataset import Ladle_labeled

def save_model(model, num):
    torch.save(model.state_dict(), 'model/cls/model_{}.pth'.format(num))

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0

    model.classifier.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)
        
        accuracy = float(num_correct)/float(num_samples)*100.0

        print('Got {} / {} with accuracy {}%'.format(num_correct, num_samples, accuracy))

    model.classifier.train()

    return accuracy

def train():
    num_epoches = 300
    batch_size = 128
    device = torch.device('cuda')

    dataset = Ladle_labeled('label.csv', 'crop_data')
    train_set, test_set = random_split(dataset, [300, 87])
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 2)

    model = Classifier()
    model.encoder.load_state_dict(torch.load('model/Encoder/model_99.pth'))
    for param in model.parameters():
        param.requires_grad = False
    #for param in model.classifier.parameters():
    #    param.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Linear(512, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(inplace = True),
        nn.Linear(100, 3)
    )
    model.to(device)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    #print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    model.eval()
    model.classifier.train()
    with open('loss.csv', 'w') as f:
        for epoch in range(num_epoches):
            losses = []

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                pre = model(data)
                loss = criterion(pre, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            total_loss = sum(losses)/len(losses)
            print('\nCost at epoch {} is :{}'.format(epoch+1, total_loss))
            f.write('{},{},'.format(epoch, total_loss))

            print('Checking accuracy on Training Set')
            train_acc = check_accuracy(train_loader, model, device)
            f.write('{},'.format(train_acc))

            print('Checking accuracy on Testing set')
            test_acc = check_accuracy(test_loader, model, device)
            f.write('{}\n'.format(test_acc))

            #save_model(model, epoch)

def test():
    confused_matrix = np.zeros((3, 3), dtype = int)
    num_correct = 0
    num_samples = 0

    batch_size = 64
    device = torch.device('cuda')

    dataset = Ladle_labeled('label.csv', 'crop_data')
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    model = Classifier()
    model.load_state_dict(torch.load('model/cls/model_99.pth'))
    model.to(device)

    #print(model)

    model.eval()
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

            y_l, p_l = y.tolist(), prediction.tolist()
            for index in range(len(y_l)):
                confused_matrix[y_l[index]][p_l[index]] += 1

            #print((prediction != y).cpu().numpy())
            #print(np.array(image_path))
            # mask = (prediction != y).cpu().numpy()
            # path, pre_ar, y_ar = np.array(image_path), np.array(prediction.cpu()), np.array(y.cpu())
            # print(path[mask], y_ar[mask], pre_ar[mask], sep = '\n')
            # print()
        
        accuracy = float(num_correct)/float(num_samples)*100.0

        print('Got {} / {} with accuracy {}%'.format(num_correct, num_samples, accuracy))
        print('Confused Matrix:')
        print(confused_matrix)

if __name__ == '__main__':
    train()
    # test()
