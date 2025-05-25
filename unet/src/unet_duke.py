import os, random
import argparse
import numpy as np
import torch
import cv2 as cv
# import glob
# from PIL import Image
from os import listdir

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.nn.modules.loss import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight

# from torchsummary import summary
import torch.nn as nn
import pytorch_unet


from collections import defaultdict
import torch.nn.functional as F
# from loss import dice_loss
from skimage import measure
from scipy.ndimage import zoom

import faulthandler
faulthandler.enable()

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='/Users/durutandogan/Desktop/COMP491/duke_original', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='contains_lesion/', help='list dir')
parser.add_argument('--num_class', type=int,
                    default=2, help='output channel of network')
# parser.add_argument('--max_iterations', type=int,
#                     default=30000, help='maximum epoch number to train')
parser.add_argument('--num_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
# parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
# parser.add_argument('--deterministic', type=int,  default=1,
#                     help='whether use deterministic training')
parser.add_argument('--lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
# parser.add_argument('--img_size', type=int,
#                     default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
# parser.add_argument('--n_skip', type=int,
#                     default=3, help='using number of skip-connect, default is num')
# parser.add_argument('--vit_name', type=str,
#                     default='R50-ViT-B_16', help='select one vit model')
# parser.add_argument('--vit_patches_size', type=int,
#                     default=16, help='vit_patches_size, default is 16')
parser.add_argument('--fold', type=int,
                    default=1, help='run number of the model')
parser.add_argument('--run', type=int,
                    default=1, help='run number of the model')
parser.add_argument('--alpha', type=float,  default=0.01,
                    help='alpha value for HausdorffDTLoss')
parser.add_argument('--warm_up_epoch', type=int,
                    default=5, help='warm_up_epoch')
parser.add_argument('--num_task', type=int,
                    default=1, help='num_task')
parser.add_argument('--input_size', type=tuple,
                    default=(224,512), help='input_size')
parser.add_argument("--loss_dice", action="store_true", 
                    help="increase output verbosity") 
args = parser.parse_args()


if __name__ == "__main__":


    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.nnpack.enabled = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]='1';

    fold = args.fold
    run = args.run
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    start_epoch = 1
    warm_up_epoch = args.warm_up_epoch
    num_class = args.num_class
    feature_map_size = 16
    alpha = args.alpha
    lr= args.lr
    num_task = args.num_task
    dataPath = args.dataset
    list_dir = args.list_dir
    loss_dice = args.loss_dice
    input_size = args.input_size

#     criterion = pytorch_unet.PHLoss(alpha)
    # criterion = pytorch_unet.PHLoss_LifeTime(alpha)
    # criterion = pytorch_unet.PHLossFPFN(alpha)

    # criterion_Dice = pytorch_unet.DiceLoss()
    criterion_DiceTunet = pytorch_unet.DiceLoss_TUnet(num_class)
    criterion_bce = CrossEntropyLoss(reduction='none')    

    model_name = 'unet_fold{}_run{}'.format(fold, run)
    # print(f'model_name: {model_name}, seed: {seed}, num_epochs: {num_epochs}, batch_size: {batch_size}, warm_up_epoch: {warm_up_epoch}, feature_map_size: {feature_map_size}, alpha: {alpha}, lr: {lr}, num_class: {num_class}, num_task: {num_task}, dataPath: {dataPath}, list_dir: {list_dir}, input_size: {input_size}, loss_dice: {loss_dice}')
    print(args)
    
    snapshot_path = 'models/'

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    transforms_applied = [transforms.ToTensor()]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    InputPath =  f'{dataPath}/image/'
    outputPath = f'{dataPath}/layer/'

    #fdmapPath = f'{dataPath}/fdmaps_center/'


    class ImageDataset(Dataset):
        def __init__(self, inputPath, outputPath, input_size, split, list_dir, transforms_=None):
            self.inputPath = inputPath
            self.outputPath = outputPath
            self.input_size = input_size
            self.transform = transforms.Compose(transforms_)
            self.names = open(os.path.join(list_dir, f'fold{fold}/{split}.txt')).readlines()

        def __getitem__(self, index):
            slice_name = self.names[index].strip('\n')
            image = cv.imread(os.path.join(self.inputPath, f'{slice_name}'), 0)
            label = cv.imread(os.path.join(self.outputPath, f'{slice_name}'), 0)
            x, y = image.shape
            if x != self.input_size[0] or y != self.input_size[1]:
#                 image = cv.resize(image, (512, 224))
#                 label = cv.resize(label, (512, 224))
                image = zoom(image, (self.input_size[0] / x, self.input_size[1] / y), order=3)
                label = zoom(label, (self.input_size[0] / x, self.input_size[1] / y), order=0)
            image = (image - image.mean()) / image.std()
            label = (label> 0).astype(np.uint8)

            golds = []           
            golds.append(torch.tensor(label))

#             for i in range(1, num_task):
#                 fd_map = np.loadtxt(fdmapPath + self.names[index] + '.fdmap' + str(i))
#                 fd_map = (fd_map - fd_map.mean()) / (fd_map.std())
#                 golds.append(torch.tensor(fd_map))

            if self.transform:
                image = self.transform(image)

            return [image, golds, slice_name]

        def __len__(self):
            return len(self.names)


    # Training data loader
    train_loader = DataLoader(
        ImageDataset(InputPath, outputPath, input_size, 'train', list_dir, transforms_=transforms_applied),
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        #num_workers=1
    )
    # Validation data loader
    val_loader = DataLoader(
        ImageDataset(InputPath, outputPath, input_size, 'val', list_dir, transforms_=transforms_applied),
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        #num_workers=1
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    def calc_loss(pred, target, metrics, device, names, epoch, phase, bce_weight=0.5, ph = False):
    #     bce = F.binary_cross_entropy_with_logits(pred, target, weights)
    #     bce = F.binary_cross_entropy_with_logits(pred, target)
    #     bce = nn.BCEWithLogitsLoss()(pred, target)
    
#         class_weights = torch.tensor([1., 2.]).to(device)
#         bce_full = nn.CrossEntropyLoss(weight=class_weights,reduction='none')(pred, target)
        bce_full = criterion_bce(pred, target)
        bce = bce_full.mean()
        
#         dice = criterion_DiceTunet(pred, target).mean()    

    #     pred = F.sigmoid(pred)
    #     pred = F.softmax(pred, dim = 1)
    #     dice = dice_loss(pred, target)

    #     loss = bce * bce_weight + dice * (1 - bce_weight)

#         if epoch>warm_up_epoch and phase == 'train':
       # if epoch>warm_up_epoch:
       #     ph_loss = criterion(bce_full, pred, target, device, names)
        #    metrics['ph'] += ph_loss.data.cpu().numpy() * target.size(0)
        #    loss = ph_loss

        if loss_dice:
            dice = criterion_DiceTunet(pred, target).mean()   
            loss = bce + dice
            metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        else:
            loss = bce

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss


    def print_metrics(metrics, epoch_samples, phase):    
        outputs = []
        new_metrics = {x: metrics[x] for x in metrics.keys() if x != 'loss'}
        for k in new_metrics.keys():
            outputs.append("{}: {:4f}".format(k, new_metrics[k] / epoch_samples))
        outputs.append("loss: {:4f}".format(metrics['loss'] / epoch_samples))
        print("{}: {}".format(phase, ", ".join(outputs)))   


    dtype = torch.FloatTensor

    def train_model(model, optimizer, scheduler, epochs):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e15
        mse_loss = nn.MSELoss()

    #     f = open('thickness.txt', 'w')
        for epoch in range(epochs[0], epochs[1]+1):
            print('\nEpoch {}/{}'.format(epoch, epochs[1]))
            print('-' * 10)

            time_elapsed = []
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                since = time.time()
                if phase == 'train':
    #                 scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group['lr'])

                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

    #             for inputs, labels in dataloaders[phase]:
                for inputs, labels, names in dataloaders[phase]:
                    inputs = inputs.type(dtype).to(device)
                    labels = [x.long().to(device) for x in labels]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
    #                     output, out_list = model(inputs)
                        output, out2 = model(inputs)
#                         print(output.shape, out2.shape)
                        loss = calc_loss(output, labels[0], metrics, device, names, epoch, phase, ph = True)
        
                        if num_task>1:
                            mse = 1e-2 * mse_loss(out2, inputs)
                            metrics['mse'] += mse.data.cpu().numpy() * inputs.size(0)
                            loss += mse
                            metrics['loss'] += mse.data.cpu().numpy() * inputs.size(0)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = (metrics['loss']) / epoch_samples

                # deep copy the model
                if phase == 'val' and epoch_loss <= best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, f'{snapshot_path}/{model_name}')

                if phase == 'val':
                    valid_loss = epoch_loss
                    scheduler.step(epoch_loss)

    #             model_wts = copy.deepcopy(model.state_dict())
    #             torch.save(model_wts, './models/' + model_name + f'_{epoch}')

                time_elapsed.append(time.time() - since)
            for phase_time, phase in zip(time_elapsed, ['train', 'val']):
                print('{}: {:.0f}m {:.0f}s'.format(phase, phase_time // 60, phase_time % 60))
        print('Best val loss: {:4f}'.format(best_loss))

    #     torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'loss': epoch_loss,
    #     'seed': seed,  
    #     'batch_size': batch_size, 
    #     'warm_up_epoch': warm_up_epoch, 
    #     'feature_map_size': feature_map_size, 
    #     'alpha': alpha, 
    #     'lr': lr
    #     }, f'./models/{model_name}.pt' )


        # load best model weights
        model.load_state_dict(best_model_wts)
    #     f.close()
        return model


    import torch.optim as optim
    from torch.optim import lr_scheduler
    import time
    import copy


    model = pytorch_unet.UNet(num_class, f_size=feature_map_size, task_no=num_task)
    # summary(model, input_size=(1, 128, 1024), device="cpu")
    model = model.to(device)

    # if cnt:
    #     model.load_state_dict(torch.load('./models/' + model_name))
    
    # model.load_state_dict(torch.load('./models/UNet_fold4_run1_40'))


    optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     optimizer_ft = optim.Adam(model.parameters(), lr=lr)
    # optimizer_ft = optim.Adadelta(filter(model.parameters()), lr=1e-1)

    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=60, gamma=0.1) 
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=20)
    
    # path = f'./models/{model_name}.pt'
    # if os.path.exists(path): 
    #     checkpoint = torch.load(path) 
    #     if (checkpoint['seed']==seed) and (checkpoint['batch_size']==batch_size) and (checkpoint['warm_up_epoch']==warm_up_epoch) and (checkpoint['feature_map_size']==feature_map_size) and (checkpoint['alpha']==alpha) and (checkpoint['lr']==lr):
    #         if num_epochs>checkpoint['epoch']:
    #             model.load_state_dict(checkpoint['model_state_dict'])
    #             optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
    #             exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])
    #             start_epoch = checkpoint['epoch']+1
    #             print(f'Resuming from the last epoch: {checkpoint['epoch']}')
    #         else:
    #             print('Model was already trained.')
    #             exit(0)    
    #     else:
    #         print('Model already exists.')
    #         exit(0)
    
    model = train_model(model, optimizer_ft, exp_lr_scheduler, epochs=(start_epoch, num_epochs))
    
    import subprocess
    p = subprocess.run(f'python test_duke.py --fold {fold} --run {run} --num_task 1 --num_class 2', shell=True)

    