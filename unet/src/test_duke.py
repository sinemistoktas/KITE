import os
import numpy as np
import torch
import cv2 as cv
from os import listdir
import argparse

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

# from torchsummary import summary
import torch.nn as nn
import pytorch_unet
from scipy.ndimage import zoom


import errno
def makeDirectory(directoryPath):
    try:
        os.mkdir(directoryPath) 
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='4'; 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='/userfiles/scansiz22/datasets/duke_original', help='experiment_name')
#parser.add_argument('--list_dir', type=str,
                    #default='floss/lists/contains_lesion', help='list dir')
parser.add_argument('--list_dir', type=str,
                    default='contains_lesion', help='list dir')

parser.add_argument('--num_class', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--fold', type=int,
                    default=1, help='run number of the model')
parser.add_argument('--run', type=int,
                    default=1, help='run number of the model')
parser.add_argument('--num_task', type=int,
                    default=1, help='num_task')
parser.add_argument('--input_size', type=tuple,
                    default=(224,512), help='input_size')
args = parser.parse_args()

if __name__ == "__main__":

    fold = args.fold
    run = args.run
    num_task = args.num_task
    batch_size = 8
    num_class = args.num_class
    feature_map_size = 16
    dataPath = args.dataset
    list_dir = args.list_dir
    input_size = args.input_size

    model_name = 'unet_fold{}_run{}'.format(fold, run)
    model_path = 'models'
    result_path = 'results'
    makeDirectory(result_path)
    makeDirectory(f'{result_path}/{model_name}')  
#     os.makedirs(f'results2/{model_name}',exist_ok=True)


    transforms_applied = [transforms.ToTensor()]

    InputPath =  f'{dataPath}/image/'
    outputPath = f'{dataPath}/lesion/'

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
            x, y = image.shape
            if x != self.input_size[0] or y != self.input_size[1]:
#                 image = cv.resize(image, (512, 224))
                image = zoom(image, (self.input_size[0] / x, self.input_size[1] / y), order=3)
            image = (image - image.mean()) / image.std()

#             golds = []
#             label = cv.imread(os.path.join(self.outputPath, f'{slice_name}.png'), 0)
#             label = (label> 0).astype(np.uint8)
#             label = cv.resize(label, (512, 224))
            
#             golds.append(torch.tensor(label))

#             for i in range(1, num_task):
#                 fd_map = np.loadtxt(fdmapPath + self.names[index] + '.fdmap' + str(i))
#                 fd_map = (fd_map - fd_map.mean()) / (fd_map.std())
#                 golds.append(torch.tensor(fd_map))

            if self.transform:
                image = self.transform(image)

            return [image, slice_name]

        def __len__(self):
            return len(self.names)

    # Test data loader
    test_loader = DataLoader(
        ImageDataset(InputPath, outputPath, input_size, 'test', list_dir, transforms_=transforms_applied),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    
    dtype = torch.FloatTensor

    model = pytorch_unet.UNet(num_class, f_size=feature_map_size, task_no=num_task)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.load_state_dict(torch.load(f'{model_path}/{model_name}'))
    model.eval()

    # names = sorted(listAllOCTFiles(tsInputPath, imagePostfix))

    for i, batch in enumerate(test_loader):
        inputs = batch[0]
        names = batch[1]
        inputs = inputs.type(dtype).to(device)
    #     output, out_list = model(inputs)
        output = model(inputs)
    #     pred = torch.sigmoid(output).squeeze(0)
        preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
    #     pred = pred.data.cpu().numpy().transpose((1, 2, 0))
        preds = preds.data.cpu().numpy()
    #     pred = (pred[:,:,0] > 0.5).astype(np.int_)
        
#         out2 = output2.squeeze().data.cpu().numpy()
        for j in range(len(preds)):
            name = names[j][:-4]
            fname = f"{result_path}/{model_name}/{name}.png"
            pred = preds[j].astype(float)
#             pred = cv.resize(pred, (512, 224))
            #cv.imwrite(fname, pred)
            cv.imwrite(fname, pred.astype(np.uint8))



#             f_name = f"results2/{model_name}/{name}"
#             np.savez(f_name, image=out2[j])

#         for j in range(1, num_task):
#             out = out_list[j-1].squeeze(0)
#             out = out.data.cpu().numpy().transpose((1, 2, 0))
#             fname_fd = "./results2/" + model_name + '/'+ names[i] + '.fdmap' + str(j)
#             np.savetxt(fname_fd, out[:,:,0])
