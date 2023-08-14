import cv2
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
provinces = {
    "北京市": [], "天津市": [], "河北省": [], "山西省": [], "内蒙古自治区": [], "辽宁省": [], "吉林省": [],
    "黑龙江省": [], "上海市": [], "江苏省": [], "浙江省": [], "安徽省": [], "福建省": [], "江西省": [],
    "山东省": [], "河南省": [], "湖北省": [], "湖南省": [], "广东省": [], "广西壮族自治区": [], "海南省": [],
    "重庆市": [], "四川省": [], "贵州省": [], "云南省": [], "西藏自治区": [], "陕西省": [], "甘肃省": [],
    "青海省": [], "宁夏回族自治区": [], "新疆维吾尔自治区": [],
}
import numpy as np
import torch.nn.functional as F

visibility = [1, 0, 1]
w_ = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T
print(w_)
w = w_[np.array(visibility) == 1].T
print(w)

cam_w_extrinsics = [[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [0, 0, 0, 1]]
print(cam_w_extrinsics[-1][-1])
torch.tensor(cam_w_extrinsics).float().cuda()
print(cam_w_extrinsics)
# maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
#                                             [-1, 0, 0, 0],
#                                             [0, -1, 0, 0],
#                                             [0, 0, 0, 1]], dtype=float)
# cam_extrinsics = cam_w_extrinsics @ maxtrix_camera2camera_w
# print(cam_extrinsics)
#
# A = torch.tensor([[1,2,1,2],[3,4,3,4],[5,6,5,6],[7,8,7,8],[9,10,9,10]])
# print(A.shape)
# print(A)
# B = A.view((5,2,2))
# print(B.shape)
# print(B)

hg = [[9.97315583e-01, -7.81008847e-19, 2.57702637e+00],
      [8.25583487e-16, 9.97315512e-01, 8.56324473e+01],
      [1.01449342e-18, -8.12895181e-22, 1.00000000e+00]]
image = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
# image = np.array(image, dtype=np.float32)
# print(image)
# h_inv = np.linalg.inv(hg)
# image_h = cv2.warpPerspective(image, h_inv, (3, 3))
# print(image_h)

#
# def warp_perspective(img, M, output_shape):
#     # 创建网格坐标
#     H, W = output_shape[0], output_shape[1]
#     y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
#     grid = torch.stack((x, y), dim=-1).float().unsqueeze(0)  # 添加批次维度
#
#     # 进行透视变换
#     M_inv = torch.inverse(M)
#     warped_grid = torch.matmul(grid, M_inv.transpose(1, 2)).squeeze()  # 移除批次维度
#
#     # 进行插值
#     warped_img = F.grid_sample(img.unsqueeze(0), warped_grid.unsqueeze(0), padding_mode='border').squeeze()
#
#     return warped_img
#
# # 假设有一个输入图像 img，透视矩阵 M 和目标图像的输出形状为 output_shape
# output_shape = (1080, 1920)
# M=torch.tensor([[[9.97315583e-01, -7.81008847e-19, 2.57702637e+00],
#       [8.25583487e-16, 9.97315512e-01, 8.56324473e+01],
#       [1.01449342e-18, -8.12895181e-22, 1.00000000e+00]]])
import torch.nn.functional as F
import torchvision.transforms.functional as TF
image = cv2.imread(r'../csrc/a.jpg')
image = TF.to_tensor(image)  # 将图片转换为张量，形状为 (C, H, W)
image = image.unsqueeze(0)  # 添加批次维度，得到形状为 (1, C, H, W)
H = torch.tensor([[0.5278606, -0.0441472, -0.36988848], [0.054597948, -0.21017244, 0.13434763], [0.28049323, 0.54171556, 1.0]], dtype=torch.float32)  # 替换为实际的单应性矩阵
H=H.unsqueeze(0)
warped_img = F.grid_sample(image, H)
# warped_img = warp_perspective(image, M, output_shape)
from PIL import Image
import matplotlib.pyplot as plt
#
# 图片路径
# img = Image.open("/home/newj/图片/space.jpeg")

plt.figure("Image")  # 图像窗口名称
plt.imshow(warped_img)
plt.axis('on')  # 关掉坐标轴为 off
plt.title('warped_img')  # 图像题目

# 必须有这个，要不然无法显示
plt.show()

id=1
id2 = 3
print('[%d]'%(id),id2)
ids=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
print(ids)

a = 1e-7
print('%.9f'%(a))