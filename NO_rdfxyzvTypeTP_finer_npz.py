from __future__ import print_function
import contextlib
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from PointNet.point_model import PointNetOneXY
from PointNet.dataset import MDtrajRDFxyzvTypeNPZ
# import EarlyStopping
from PointNet.pytorchtools import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def load_data(data_dir): 
    train_set = MDtrajRDFxyzvTypeNPZ(
        root=data_dir)
    
    val_set = MDtrajRDFxyzvTypeNPZ(
        root=data_dir,
        split='val')

    test_set = MDtrajRDFxyzvTypeNPZ(
        root=data_dir,
        split='test')
    return train_set, val_set, test_set 

def train(train_loader, model, optimizer, epoch, device):
    """ Train the model on num_steps batches 
    Args: 
        train_loader: a torch.utils.data.DataLoader object that fetches the data
        model: the neural network 
        optimizer: adams 
    """
    model.train()
    running_loss = 0.0
    num_batch = len(train_loader)

    for i, (points, target, sys, frame) in enumerate(train_loader): 
        points = points.transpose(2, 1) 
        points, target, sys, frame = points.to(device), target.to(device), sys.to(device), frame.to(device)
        # zero the paramter gradients 
        optimizer.zero_grad()

        # forward + backward + optimize 
        pred = model(points, sys)

        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        # print statistics 
        running_loss += loss.item()
        print('[%d: %d/%d] train loss: %f ' % (epoch, i, num_batch, loss.item()))

    return running_loss / num_batch 

def validate(val_dataloader, model, device): 
    model.eval()
    val_running_loss = 0.0
    num_batch = len(val_dataloader)

    with torch.no_grad(): 
        for points, target, sys, frame in val_dataloader:
            points = points.transpose(2, 1)
            points, target, sys, frame = points.to(device), target.to(device), sys.to(device), frame.to(device)

            outputs = model(points, sys)

            loss = F.mse_loss(outputs, target)
            val_running_loss += loss.item() 

    return val_running_loss / num_batch
        
def plot_pramas(test_y, test_pred, foldername, filename): 
    # print("R2 of training is: ", r2_score(train_y, train_pred))
    print("R2 of test is: ", r2_score(test_y, test_pred[:, 3:]))

    np.savetxt(f'{foldername}/predict_{filename}.txt', test_pred)
    np.savetxt(f'{foldername}/target_{filename}.txt', test_y)

    test_mse = mean_squared_error(test_y, test_pred[:,3:])
    test_mae = mean_absolute_error(test_y, test_pred[:,3:])

    print('Test set results for %i samples:' % test_pred.shape[0])
    print('MSE:', test_mse)
    print('MAE:', test_mae)

def test_accuracy(net, testloader, foldername, filename, device): 
    test_pred = []
    test_y = [] 

    running_loss = 0
    with torch.no_grad():
        for data in testloader:
            points, target, sys, frame = data
            points = points.transpose(2, 1) 
            points, target, sys, frame = points.to(device), target.to(device), sys.to(device), frame.to(device)
            outputs = net(points, sys)
            loss = F.mse_loss(outputs, target)

            running_loss += loss.item() * points.size(0)

            pred_val_numpy = outputs.data.cpu().numpy()
            target_val_numpy = target.data.cpu().numpy()
            # test_pred.append(pred_val_numpy)
            test_pred.append(np.concatenate([sys.data.cpu().numpy(), frame.data.cpu().numpy(), pred_val_numpy],axis = 1))
            test_y.append(target_val_numpy)

    test_pred = np.concatenate(test_pred, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    plot_pramas(test_y, test_pred, foldername, filename)
    print('MSE loss on test set is:', running_loss / len(testloader.dataset))

def train_model(model, device, train_loader, val_loader, test_loader, optimizer, lr_scheduler, isSch, res_dir, name, patience = 20, n_epochs = 100): 
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 

    blue = lambda x: '\033[94m' + x + '\033[0m'
    # initialize the early_stopping object
    checkpoint_dir = os.path.join(res_dir, 'checkpoints')

    with contextlib.suppress(OSError):
        os.makedirs(res_dir)
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'{name}.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path = checkpoint_path)

    for epoch in tqdm(range(1, n_epochs + 1)):
        ###################
        # train the model #
        ###################
        train_epoch_loss = train(train_loader, model, optimizer, epoch, device)
        val_epoch_loss = validate(val_loader, model, device)

        if isSch: 
            lr_scheduler.step(val_epoch_loss)

        # print loss every epoch 
        print('[%d] train loss: %f ' % (epoch, train_epoch_loss))
        print('[%d] %s loss: %f' % (epoch, blue('validate'), val_epoch_loss))

        avg_train_losses.append(train_epoch_loss)
        avg_valid_losses.append(val_epoch_loss)

        # add early stopping 
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop: 
            print("Early stopping")
            break 

    np.savetxt(os.path.join(res_dir,f'train_loss_{name}.csv'), avg_train_losses)
    np.savetxt(os.path.join(res_dir,f'val_loss_{name}.csv'), avg_valid_losses)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))

    return test_accuracy(model, test_loader, res_dir, name, device)

def train_main(): 
    data_dir = "/pscratch/sd/c/chunhui/RDF_rerun/NO_dataset/2000frames"
    config = {
        "batch_size": 256,
        "dropout": 0,
        "l1": 64,
        "l2": 128,
        "l3": 1024,
        "l4": 1024,
        "l5": 2048,
        "lr": 0.001,
        "isSch" : False,
        "addTP" : True,
        "isBN" : False,
        "isLN" : False, 
        "pooling" : 'max'
    }

    # get dataset 
    train_set, val_set, test_set = load_data(data_dir)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0)

    val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(
            test_set, 
            batch_size=int(config["batch_size"]), 
            shuffle=False, num_workers=0)

    name = f'NO_xyzvType_addTP{config["addTP"]}_batch{config["isBN"]}'
    res_dir = '/pscratch/sd/c/chunhui/RDF_rerun/res/NO/2000frames'
    
    model = PointNetOneXY(config["l1"], config["l2"], config["l3"], config["l4"], config["l5"], 
                            config["dropout"], config["isBN"], config["isLN"], config["addTP"], config["pooling"], "NO")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda')
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    train_model(model, device, train_loader, val_loader, test_loader, 
                               optimizer, lr_scheduler, config["isSch"], res_dir, name, 20, 1000)

def predict_on_test():
    # data_dir = "/data/chunhui_backup/NO/data/train_val_test_split"
    data_dir = "/data/chunhui_backup/NO/data/test_compute_efficiency"
    config = {
        "batch_size": 256,
        "dropout": 0,
        "l1": 64,
        "l2": 128,
        "l3": 1024,
        "l4": 1024,
        "l5": 2048,
        "lr": 0.001
    }

    # get dataset 
    _, test_set, _ = load_data(data_dir)
    
    test_loader = torch.utils.data.DataLoader(
            test_set, 
            batch_size=int(config["batch_size"]), 
            shuffle=False, num_workers=0)

    addTP = False # False
    batch = True
    name = f'NO_xyzvType_addTP{addTP}_batch{batch}'
    res_dir = '/data/chunhui_backup/RDF_finer_rerun/NO'
    checkpoint_dir = os.path.join(res_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, f'{name}.pt')
    
    model = PointNetRegxyzvType(config["l1"], config["l2"], config["l3"], config["l4"], config["l5"], config["dropout"], addTP, batch)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda')
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))
    last_val_loss = test_accuracy(model, test_loader, res_dir, f'{name}_test', device)

    print("The final test loss: ", last_val_loss)

if __name__ == '__main__': 
    import time 
    start_time = time.perf_counter()
    train_main() 
    # predict_on_test()
    end_time = time.perf_counter() 
    print('time used to train model with 10 patience is: ', (end_time - start_time), 'seconds')
    

    