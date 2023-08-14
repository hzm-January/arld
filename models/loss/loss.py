import torch
from torch import Tensor, nn as nn


class HMSELoss(nn.Module):
    def __int__(self):
        super(HMSELoss, self).__init__()
        self.mse_sum_loss = nn.MSELoss(reduction='sum')
        self.mse_loss = nn.MSELoss()
        self.sl1_loss = nn.SmoothL1Loss()

    def forward_bk(self, outputs, targets):
        # 求targets中不为0元素个数
        num_nonzero_gt = torch.sum(targets != 0, dtype=torch.float32).item()
        num_nonzero_gt = 1 if num_nonzero_gt == 0 else num_nonzero_gt
        # 求MSE，reduction='sum'
        mse_loss = self.mse_loss(outputs, targets)
        return mse_loss / num_nonzero_gt

    def forward(self, featmap, gt):
        assert (featmap.shape[1:] == gt.shape[1:])
        batch_size = featmap.shape[0]
        C = gt[gt < self.ignore_label].max().item()
        loss = 0.0
        for b in range(batch_size):  # featmap.shape[0] is batch size
            bfeat = featmap[b][0]  # 取出第一张图像预测结果数据
            bgt = gt[b][0]  # 取出第一张图像的标签
            # 将两个二维张量展平为一维张量
            bfeat = bfeat.view(-1)
            bgt = bgt.view(-1)
            for i in range(1, int(C) + 1):
                instance_mask = bgt == i  # 每条车道线上的元素都用i表示，取出所有等于i的像素值，都标记为true，其余像素值标记为false，相当于提取一条编号为i的车道线。
                if instance_mask.sum() == 0:
                    continue  # 如果bgt中所有像素都与i不相等，跳出，提取等于i+1的下一条车道线的像素值
                # pos_mask = bgt[:,
                #            instance_mask].T.contiguous()  # mask_num x N 选出所有通道中在instance_mask对应位置为true的所有像素，相当于提取当前车道线的所有标记点的位置信息。
                # 对当前车道线，进行损失计算。
                loss += self.mse_loss(bfeat[instance_mask], bgt[instance_mask])
        return loss
if __name__ == '__main__':
    t = torch.tensor([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    num_t = torch.sum(t != 0).item()
    print(num_t)
    a = torch.rand((8, 1, 2, 3))
    b = torch.rand((8, 1, 2, 3))
    print(len(a))
    print(zip(a, b))
