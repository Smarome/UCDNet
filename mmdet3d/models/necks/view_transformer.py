# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
import json

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS



@NECKS.register_module()
class LSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        GD = 11,
        accelerate=False,
        sid=False,
        collapse_z=True,
    ):
        super(LSSViewTransformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)  # 创建BEV_grid
        self.sid = sid
        self.GD = GD
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)  # [99,16,44,3],3中存着每个块里存储宽度上第几个像素，高度上第几个像素，深度的数值
        self.out_channels = out_channels  # 80
        self.in_channels = in_channels  # 256
        self.depth_net = nn.Conv2d(
            in_channels, 1 + self.GD + self.out_channels, kernel_size=1, padding=0)  # %%%多了一个1，用来算object_pred
        # self.depth_net = nn.Conv2d(
        #     in_channels, 1 + self.D + self.out_channels, kernel_size=1, padding=0)  # %%%多了一个1，用来算object_pred
        self.accelerate = accelerate  # False
        self.initial_flag = True
        self.collapse_z = collapse_z  # True

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])  # tensor([-51.2000, -51.2000,  -5.0000])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])  # tensor([0.8000, 0.8000, 8.0000])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])  # tensor([128., 128.,   1.])

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature
                 size.
        """

        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample  # 16,44

        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)  # [99,16,44],d[0]是1，d[98]全是99

        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)  # [99,16,44] x[n][m]=[0,...,703]一共44个数
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)  # # [99,16,44] x[n][][m]=[0,...,255]一共16个数

        # D x H x W x 3  99,16,44,3 每个块里存储宽度上第几个像素，高度上第几个像素，深度的数值
        return torch.stack((x, y, d), -1)

    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape  # B:12, N:6

        # post-transformation，图像预处理逆变换，3里存的是每个格对应的真实图像2.5D坐标
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)  # 12 6 99 16 44 3,减去平移，frustum维度改变是因为广播机制
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))  # 乘以图像旋转矩阵的逆矩阵，最后添加一个大小为1的新维度，这个1仅仅是把最后一位中的三个数都加了个括号而已
        # with open('img_inaug.pickle', 'wb') as f:
        #     pickle.dump(points, f)

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)  # 输出这个还是底下得points对同名点得3维位置不影响
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):  # coor: [12,6,99,16,44,3]  depth: [12,6,99,16,44]  feat: [12,6,80,16,44]
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')  # %%% 原来如此
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)  # [12,6,16,44,80]
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C) 12,1 128,128,80
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape  #  [12,6,99,16,44,3]
        num_points = B * N * D * H * W  # 总共有这么多个点，5018112
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)  # 一共有这么多深度[0,1,2,...,5018111]
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)  # 一共有这么多特征向量[0,...,50687]
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)  # [12,6,1,16,44]
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()  # 5018112个数，同图上的特征向量投影到各个深度上是相同的，因此无需对深度排序
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))  # 用于将张量转换到与另一个张量（在这个例子中是 coor）相同的数据类型和设备上,最后一维
        coor = coor.long().view(num_points, 3)  # [5018112,3]，变整数了，得到BEV坐标系下的坐标，不是ego，而是BEV坐标系
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)  # 每个点属于哪个batchsize[0,0,...,1,1,...,11,11]
        coor = torch.cat((coor, batch_idx), 1)  # [5018112,4]

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])  # 5018112
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]  # BEV下的视锥点，高度维全是0了[884293,4]，和ranks_depth以及ranks_feat的对应关系还存在
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])  # 计算视锥点全局索引，第一个batch的点全是0，第11个batch的点全是180224
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]  # 体素编号，说实话我还是理解不了
        order = ranks_bev.argsort()  # bev下，全局索引为相同的值排在一起，这个order是个啥
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]  # # 错位比较，可以使得索引位置相同的，只有最后一个位置为True
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)  # 每个为True的索引位置，向前累加的长度
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat, object_tensor):
        B, N, C, H, W = input[0].shape  # [12,6,256,6,44]
        x = input[0]
        object_tensor_pred = object_tensor[:B*N, :, :].view(B, N, H, W)
        # object_tensor_gt = object_tensor[B*N: , :, :].view(B, N, H, W)
        object_tensor_pred_expanded = object_tensor_pred.unsqueeze(2)
        # object_tensor_gt_expanded = object_tensor_gt.unsqueeze(2)
        object_tensor_pred_expanded_bin_1 = (object_tensor_pred_expanded > 0.7).float()  # 先改了吧本来是0.7
        # object_tensor_pred_expanded_bin_2 = (object_tensor_pred_expanded > 0.45).float()
        # object_tensor_gt_expanded_bin = (object_tensor_gt_expanded > 0.7).float()
        feature = x * object_tensor_pred_expanded_bin_1
        if self.training:
            most_corresponding_points = self.find_most_corresponding_points(feature, threshold=0.7)  # %%%
        else:
            most_corresponding_points = []
        depth = depth.view(B, N, self.D, H, W)
        # with open('depth.pickle', 'wb') as f:
        #     pickle.dump(depth, f)
        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            # depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(*input[1:7])  # [12,6,99,16,44,3],特征图的特征像素投影成的点云点在世界坐标系的位置
            # with open('coor.pickle', 'wb') as f:
            #     pickle.dump(coor, f)
            location1 = []
            location2 = []
            if self.training and len(most_corresponding_points) > 0:
                for p in most_corresponding_points:
                    pos1 = coor[p[0], p[1], :, p[2], p[3], :]
                    pro1 = depth[p[0], p[1], :, p[2], p[3]]
                    pos2 = coor[p[0], p[4], :, p[5], p[6], :]
                    pro2 = depth[p[0], p[4], :, p[5], p[6]]
                    loc1 = torch.matmul(pos1.T, pro1)
                    loc2 = torch.matmul(pos2.T, pro2)
                    location1.append(loc1)
                    location2.append(loc2)

            location = [location1, location2]
            # %%%
            # tran_feat_sample = tran_feat.view(B, N, self.out_channels, H, W) * object_tensor_pred_expanded_bin_2  # 被二维预测mask加权后的，如果不加权可以去掉*object_tensor_pred
            tran_feat_sample = tran_feat.view(B, N, self.out_channels, H, W)
            bev_feat = self.voxel_pooling_v2(
                coor, depth, tran_feat_sample)  # 12 80 128 128  depth 12*6 59 16 44
            # with open('bev_feat.pickle', 'wb') as f:
            #     pickle.dump(bev_feat, f)
        return bev_feat, location, object_tensor  # %%%

    def view_transform(self, input, depth, tran_feat, object_tensor):
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat, object_tensor)

    def forward(self, input):  # 重点从这里开始
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        x = input[0]# [12, 6, 256, 16, 44]
        B, N, C, H, W = x.shape  # H=256/16=16, W=704/16=44
        x = x.view(B * N, C, H, W)  # [72,256,16,44]这样的话是意味着前6个在第一个batchsize里面
        x = self.depth_net(x)  # 72,179,16,44   179=99（深度数）+80（通道数），就一个卷积把特征和深度都卷了

        tran_feat = x[:, self.GD + 1:self.GD + self.out_channels + 1, ...]  # [72，80 16，44]
        # tran_feat = x[:, self.D + 1:self.D + self.out_channels + 1, ...]  # %%%[72，80 16，44]

        # tran_feat = torch.ones_like(tran_feat)  # %%%这步好像是在进行Fsores

        # 处理创新点3，概率mask
        object_pred = x[:, 0, ...]
        object_pred = self.z_score_normalize(object_pred)
        object_pred = object_pred.sigmoid()
        # with open('object_pred_new.pickle', 'wb') as f:
        #     pickle.dump(object_pred, f)
        object_mask = input[8].view(B * N, H, W)
        object_tensor = torch.cat((object_pred, object_mask), dim=0)

        # 处理创新点2，地面深度估计
        depth_digit = x[:, 1:self.GD + 1, ...]  # [72，99，16，44]每个格里储存着特征像素属于某一深度的概率
        depth = depth_digit.softmax(dim=1)
        depth = depth.view(B, N, self.GD, H, W)  # 把预测的小深度转成5维
        # 随机生成A和B张量
        depth_gd = input[7].to(torch.int64)  # 整数，别忘了
        # 初始化H张量
        depth_final = torch.zeros(B, N, self.D, H, W).to(x)
        # 反转B中的概率顺序，以匹配正确的深度顺序
        reversed_B = depth.flip(2)  # 在深度通道上反转
        # 计算真实深度的索引
        expanded_A = (depth_gd.unsqueeze(2)).to(x) - (self.GD - 1 - torch.arange(self.GD).view(1, 1, self.GD, 1, 1)).to(x)
        expanded_A = torch.clamp(expanded_A, 0, self.D - 1)  # 保证索引在有效范围内
        # 使用高效的批量操作填充H张量
        for c in range(self.GD):
            depth_index = expanded_A[:, :, c, :, :].to(torch.int64)
            depth_final.scatter_add_(2, depth_index.unsqueeze(2), reversed_B[:, :, c:c + 1, :, :])
        depth_final = depth_final.view(B * N, self.D, H, W)  # 把最终的H转会4维

        # # 原始深度估计
        # depth_digit = x[:, 1:self.D + 1, ...]
        # depth_final = depth_digit.softmax(dim=1)

        # 将地面深度直接作为深度
        # depth_gd = depth_gd.to(x)
        # depth_gd = depth_gd.to(torch.long)
        # depth = torch.zeros(B, N, self.D, H, W).to(x)
        # for b in range(B):
        #     for n in range(N):
        #         for h in range(H):
        #             for w in range(W):
        #                 value = depth_gd[b, n, h, w]
        #                 depth[b, n, value, h, w] = 1.0
        # depth_final = depth.view(B * N, self.D, H, W)  # 把最终的H转会4维
        #

        return self.view_transform(input, depth_final, tran_feat, object_tensor)  # 转换到BEV空间


    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None

    def find_most_corresponding_points(self, features, threshold=0.9):
        B, N, C, H, W = features.shape
        most_corresponding_points = []

        # 对整个特征张量进行归一化
        features_flat = features.view(B, N, C, -1).permute(0, 1, 3, 2)  # [B, N, H*W, C]
        features_flat = F.normalize(features_flat, dim=3)

        for b in range(B):
            for i in range(N):
                img1_flat = features_flat[b, i]  # [H*W, C]
                j = 0 if i == N - 1 else i + 1
                img2_flat = features_flat[b, j]  # [H*W, C]
                similarity = torch.matmul(img1_flat, img2_flat.T)  # [H*W, H*W]
                max_sim, max_indices = torch.max(similarity, dim=1)

                valid_indices = max_sim >= threshold
                max_sim = max_sim[valid_indices]
                max_indices = max_indices[valid_indices]
                idxs = torch.arange(H * W, device=features.device)[valid_indices]

                for idx, max_idx, max_sim_value in zip(idxs, max_indices, max_sim):
                    h1, w1 = divmod(idx.item(), W)
                    h2, w2 = divmod(max_idx.item(), W)
                    most_corresponding_points.append([b, i, h1, w1, j, h2, w2, max_sim_value.item()])

        most_corresponding_points = sorted(most_corresponding_points, key=lambda x: x[-1], reverse=True)[:200]  # 200
        return most_corresponding_points

    def z_score_normalize(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        normalized_tensor = (tensor - mean) / std
        return normalized_tensor



class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 bias=0.0,
                 aspp_mid_channels=-1):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        if stereo:
            depth_conv_input_channels += depth_channels
            downsample = nn.Conv2d(depth_conv_input_channels,
                                    mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for stage in range(int(2)):
                cost_volumn_net.extend([
                    nn.Conv2d(depth_channels, depth_channels, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(depth_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            self.bias = bias
        depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample),
                           BasicBlock(mid_channels, mid_channels),
                           BasicBlock(mid_channels, mid_channels)]
        if use_aspp:
            if aspp_mid_channels<0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        frustum = metas['frustum']
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)
        neg_mask = points[..., 2, 0] < 1e-3
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points = points[..., :2, :] / points[..., 2:3, :]

        points = metas['post_rots'][...,:2,:2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][...,:2].view(B, N, 1, 1, 1, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)
        grid = grid.view(B * N, D * H, W, 2)
        return grid

    def calculate_cost_volumn(self, metas):
        prev, curr = metas['cv_feat_list']
        group_size = 4
        _, c, hf, wf = curr.shape
        hi, wi = hf * 4, wf * 4
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)

        prev = prev.view(B * N, -1, H, W)
        curr = curr.view(B * N, -1, H, W)
        cost_volumn = 0
        # process in group wise to save memory
        for fid in range(curr.shape[1] // group_size):
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            wrap_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)
            cost_volumn += cost_volumn_tmp
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        return cost_volumn

    def forward(self, x, mlp_input, stereo_metas=None):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        if not stereo_metas is None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)
            cost_volumn = self.cost_volumn_net(cost_volumn)
            depth = torch.cat([depth, cost_volumn], dim=1)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class DepthAggregation(nn.Module):
    """pixel cloud feature extraction."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = checkpoint(self.reduce_conv, x)
        short_cut = x
        x = checkpoint(self.conv, x)
        x = short_cut + x
        x = self.out_conv(x)
        return x


@NECKS.register_module()
class LSSViewTransformerBEVDepth(LSSViewTransformer):

    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D, **depthnet_cfg)

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input, stereo_metas=None):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        # (x, rots, trans, intrins, post_rots, post_trans, bda) = input[:7]
        # mlp_input = input[9]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input, stereo_metas)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)

        object_tensor = torch.ones(12,16,44).cuda()

        # bev_feat, depth, _ = self.view_transform(input, depth, tran_feat, object_tensor)
        bev_feat, depth, _ = self.view_transform(input, depth, tran_feat)
        return bev_feat, depth


@NECKS.register_module()
class LSSViewTransformerBEVStereo(LSSViewTransformerBEVDepth):

    def __init__(self,  **kwargs):
        super(LSSViewTransformerBEVStereo, self).__init__(**kwargs)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=4)

@NECKS.register_module()
class LSSViewTransformer_original(LSSViewTransformer):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        GD = 11,
        accelerate=False,
        sid=False,
        collapse_z=True,
    ):
        super(LSSViewTransformer_original, self).__init__(
            grid_config = grid_config,
            input_size = input_size,
            downsample=downsample,
            in_channels=in_channels,
            out_channels=out_channels,
            GD = GD,
            accelerate=accelerate,
            sid=sid,
            collapse_z=collapse_z,)
        self.depth_net = nn.Conv2d(
            in_channels, 1 + self.D + self.out_channels, kernel_size=1, padding=0)  # %%%多了一个1，用来算object_pred

    def forward(self, input):  # 重点从这里开始
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        x = input[0]  # [12, 6, 256, 16, 44]
        B, N, C, H, W = x.shape  # H=256/16=16, W=704/16=44
        x = x.view(B * N, C, H, W)  # [72,256,16,44]这样的话是意味着前6个在第一个batchsize里面
        x = self.depth_net(x)  # 72,179,16,44   179=99（深度数）+80（通道数），就一个卷积把特征和深度都卷了

        tran_feat = x[:, self.D + 1:self.D + self.out_channels + 1, ...]  # %%%[72，80 16，44]

        # 处理创新点3，概率mask
        object_pred = x[:, 0, ...]
        object_pred = self.z_score_normalize(object_pred)
        object_pred = object_pred.sigmoid()
        # with open('object_pred_new.pickle', 'wb') as f:
        #     pickle.dump(object_pred, f)
        object_mask = input[8].view(B * N, H, W)
        object_tensor = torch.cat((object_pred, object_mask), dim=0)

        # 原始深度估计
        depth_digit = x[:, 1:self.D + 1, ...]
        depth_final = depth_digit.softmax(dim=1)

        return self.view_transform(input, depth_final, tran_feat, object_tensor)  # 转换到BEV空间

