from __future__ import print_funciton, division #python 3에서 쓰는걸 lower version 에서도 가져올 수 있도록 
import os #file o/s
from PIL import Image #pillow 에서 이미지 불러오는 module
import numpy as np 
from torch.utils.data import Dataset 
from mypath import Path 
import tqdm
from dataloaders import custom_transforms as tr
from dataloaders.utils import decode_segmap
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class PascalVOC(Dataset):
	def __init__(self, base_dir=Path.db_root_dir('pascal'), split='train', transform=None)
		'''
		Parameters
		- base_dir = path to VOC dataset directory 
		- split = train set / validation set 
		- transform: transform? 
		'''
		super().__init__()
		self._base_dir = base_dir
		self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
		self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
		self.transform = transform
		if isinstance(split, str): #if split is a string
			self.split = [split]
		else: 
			split.sort()
			self.split = split 

		_splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
		
		self.im_ids = []
		self.images = []
		self.categories = []

		for splt in tqdm(self.split): 
			with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
				lines = f.read().splitlines()

			for ii, line in enumberate(lines):
				_image = os.path.join(self._image_dir, line+ ".jpg")
				_cat = os.path.join(self._cat_dir, line + ".png")
				assert os.path.isfile(_image)
				assert os.path.isfile(_cat)
				self.im_ids.append(line)
				self.images.append(_image)
				self.categories.append(_cat)

		assert (len(self.images) == len(self.categories))
		print('Images in {}: {:d}'.format(split, len(self.images)))

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		_img, _target = self._make_img_gt_point_pair(index)
		sample = {'image': _img, 'label': _target}
		if self.transform is not None: 
			sample = self.transform(sample)
		return sample 

	def _make_img_gt_point_pair(self, index):
		#Read image and target 
		_img = Image.open(self.images[index].convert('RGB'))
		_target = Image.open(self.categories[index])
		return _img, _target

	def __str__(self):
		return 'VOC2012(split=' + str(self.split) + ')'

if __name__ == '__main__':
	composed_transforms_tr = transforms.Compose([tr.RandomHorizontalFlip(), tr.RandomSized(512), tr.RandomRotate(15), tr.ToTensor()])

	voc_train = PascalVOC(split='train', transform=composed_transforms_tr)
	dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=2)

	for ii, sample in tqdm(enumberate(dataloader)):
		for jj in tqdm(range(sample["image"].size()[0])):
			img = sample['image'].numpy()
			gt = sample['label'].numpy()
			tmp = np.array(get[jj]).astype(np.uint8)
			tmp = np.squeeze(tmp, axis=0)
			segmap = decode_segmap(tmp, dataset = 'pascal')
			img_tmp = np.transpose(img[jj], axes=[1,2,0]).astype(np.uint8)
			plt.figure()
			plt.title('display')
			plt.subplot(211)
			plt.imshow(im_tmp)
			plt.subplot(212)
			plt.imshow(segmap)
		if ii == 1: 
			break 
	plt.show(block=True)



































