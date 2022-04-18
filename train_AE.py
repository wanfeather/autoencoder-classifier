import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unet import UNet
from dataset import Ladle

def save_model(model, model_sep, num):
    torch.save(model, 'model/{}/model_{}.pth'.format(model_sep, num))

def main():
    num_epoches = 100
    lr = 1e-4
    batch_size = 64
    num_batch = int(4303/batch_size) + 1
    device = torch.device('cuda')

    dataset = Ladle('data.csv', 'crop_data')
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

    model = UNet()
    model = model.to(device)
    model = nn.DataParallel(model, device_ids = [0, 1])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    model.train()
    with open('loss.csv', 'w') as f:
        for epoch in range(num_epoches):
            losses = []

            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                target_pre = model(data)
                loss = criterion(target_pre, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                print('Epoch:{}/{}, Batches:{}/{}  {}%, Training Loss:{}'.format(epoch+1, num_epoches, batch_idx+1, num_batch, int((batch_idx+1)/num_batch*100), loss), end = '\r')

            total_loss = sum(losses)/len(losses)
            print('\nTotal Loss:{}'.format(total_loss))
            print('Saving model...')
            save_model(model.module.encoder.state_dict(), 'Encoder', epoch)
            save_model(model.module.decoder.state_dict(), 'Decoder', epoch)
            print('Model saved.')
            f.write('{},{}\n'.format(epoch, total_loss))

if __name__ == '__main__':
    main()
