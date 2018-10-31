import os
import torch
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from dataloaders.utils import recursive_glob, decode_segmap
from mypath import Path
import json
from __future__ import print_function, division
import scipy.io



class CityscapesSegmentation(data.Dataset):

    def __init__(self, root=Path.db_root_dir('cityscapes'), split="train", transform=None):

        self.root = root
        self.split = split
        self.transform = transform
        self.files = {}
        self.n_classes = 19

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.transform:
            sample = self.transform(sample)
        return sample


    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


if __name__ == '__main__':
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomScale((0.5, 0.75)),
        tr.RandomCrop((512, 1024)),
        tr.RandomRotate(5),
        tr.ToTensor()])

    cityscapes_train = CityscapesSegmentation(split='train',
                                transform=composed_transforms_tr)

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            tmp = np.squeeze(tmp, axis=0)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break
    plt.show(block=True)


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]



def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]



def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou


