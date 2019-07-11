import argparse
import os
import numpy as np
from tqdm import tqdm

from modeling import *

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloaders import test_transforms as tr
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from PIL import Image
import cv2

import torch.nn.functional as F

length = 0
width = 0
num_classes = 4
#需要与训练中的设置一致
unit_size = 512

class Test_dataset(Dataset):
	def __init__(self, unit_size):
		super().__init__()
		# 修改为测试集的路径，当前只支持一次处理一张，欢迎改进
		self.img_dir = '/data/dingyifeng/jingwei/jingwei_round1_test_a_20190619/image_4.png'
		self.unit_size = unit_size
		self.stride = int(unit_size/3)
		self.composed_transforms = transforms.Compose([
			tr.FixScaleCrop(crop_size=unit_size),
			tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			tr.ToTensor()])
		Image.MAX_IMAGE_PIXELS = 100000000000

		test_img = Image.open(self.img_dir)
		test_img = np.asarray(test_img)
		print("test_img:" + str(test_img.shape))

		global length, width
		length, width = test_img.shape[0], test_img.shape[1]

		x1, x2, y1, y2 = 0, self.unit_size ,0 ,self.unit_size
		Img = [] # 保存小图的数组
		x1_ = []
		y1_ = []
		while(x1 < length):
			#判断横向是否越界
			if x2 > length:
				x2 , x1 = length , length - self.unit_size

			while(y1 < width):
				if y2 > width:
					y2 , y1  = width , width - self.unit_size
				im = test_img[x1:x2, y1:y2, :]
				Img.append(im[:,:,0:3])   # 添加小图
				x1_.append(x1)
				y1_.append(y1)

				if y2 == width: break

				y1 += self.stride
				y2 += self.stride

			if x2 == length: break

			y1, y2 = 0, self.unit_size
			x1 += self.stride
			x2 += self.stride
		self.Img = np.array(Img) # (n, unit_size, unit_size, 3)
		self.x_ = x1_ # (n)
		self.y_ = y1_ # (n)

		assert (len(self.Img) == len(self.x_) == len(self.y_))

		# Display stats
		print('Number of images for test: {:d}'.format(len(self.Img)))

	def __len__(self):
		return len(self.Img)

	def __getitem__(self, index):
		img = Image.fromarray(self.Img[index])
		x_ = self.x_[index] 
		y_ = self.y_[index] 
		return x_, y_, self.composed_transforms(img)


test_set = Test_dataset(unit_size=unit_size)
test_loader = DataLoader(test_set, batch_size=24, shuffle=False, num_workers=8)


# load model
model = UNetResNet34().cuda()
# model = UNetSimple().cuda()


# resume
# 修改为保存模型参数文件的路径
resume_path = '/home/dingyifeng/pytorch-jingwei-master/run/jingwei/UNetResNet34/22_5/checkpoint.pth.tar'
if not os.path.isfile(resume_path):
    raise RuntimeError("=> no checkpoint found at '{}'" .format(resume_path))
checkpoint = torch.load(resume_path)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' ".format(resume_path))

model = torch.nn.DataParallel(model)


pred_total = np.zeros((num_classes, length, width))
model.eval()
for i, sample in enumerate(test_loader):
	x = sample[0] #(batch, 1)
	y = sample[1]
	img = sample[2].cuda() #(batch, 3, unit_size, unit_size)
	with torch.no_grad():
		output = model(img) #(batch, num_classes, unit_size, unit_size)
	# import pdb
	# pdb.set_trace()
	output = F.softmax(output, 1)

	output = output.data.cpu().numpy()
	for n in range(len(output)):
		x_ = x[n]
		y_ = y[n]
		pred_total[:, x_:x_+unit_size, y_:y_+unit_size] = pred_total[:, x_:x_+unit_size, y_:y_+unit_size] + output[n]

pred_out = np.argmax(pred_total, axis=0).squeeze() #(length, width)

pred_out_ = Image.fromarray(np.uint8(pred_out))
pred_out_.save("image_4_predict.png", "PNG")

# import pdb
# pdb.set_trace()


# def visualization(pred_out):
# 	# visualization
# 	B = pred_out.copy()   # 蓝色通道
# 	B[B == 1] = 255
# 	B[B == 2] = 0
# 	B[B == 3] = 0
# 	B[B == 0] = 0

# 	G = pred_out.copy()   # 绿色通道
# 	G[G == 1] = 0
# 	G[G == 2] = 255
# 	G[G == 3] = 0
# 	G[G == 0] = 0

# 	R = pred_out.copy()   # 红色通道
# 	R[R == 1] = 0
# 	R[R == 2] = 0
# 	R[R == 3] = 255
# 	R[R == 0] = 0

# 	anno_vis = np.dstack((B,G,R))
# 	anno_vis = cv2.resize(anno_vis, None, fx= 0.1, fy=0.1)
# 	cv2.imwrite('./visual/test4.png', anno_vis)

# visualization(pred_out)