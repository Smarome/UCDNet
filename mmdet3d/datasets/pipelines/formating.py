# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC
import pickle
from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.pipelines import to_tensor
from ..builder import PIPELINES
import torch
import torch.nn.functional as F
import time
@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = DC(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DC(
                    to_tensor(results['gt_bboxes_3d']))

        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect3D(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """

        RT_1, _, K, post_rots, post_trans, bda = results['img_inputs'][1:7]
        sample = self.get_ground_depth(RT_1, K, post_rots, post_trans)  # [12,16,44,2]
        Ba, h, w, c= sample.shape
        sample = sample[: int(Ba/2), :, :, :]  # 这是采样值是有可能大于1600和900的
        K = K[: int(Ba/2), :, :]
        RT_1 = RT_1[: int(Ba/2), :, :]
        gt_boxes_bda = results['gt_bboxes_3d'].data.corners
        object_images = self.project_3d_boxes_to_multiple_images(gt_boxes_bda, K, RT_1, bda)  # 我懂了，有的车在相机背面，所以呈现出来特别怪
        object_images_sample = self.sample_images_tensorized(object_images, sample)
        object_images_sample = F.max_pool2d(object_images_sample.float(), kernel_size=8)
        img_inputs_old = results['img_inputs']
        results['img_inputs'] = img_inputs_old + (object_images_sample,)
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]

        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'

    def create_frustum(self, depth_cfg, input_size, downsample):
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample  # 16,44
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)  # [99,16,44],d[0]是1，d[98]全是99
        D = d.shape[0]
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(D, H_feat, W_feat)  # [99,16,44] x[n][m]=[0,...,703]一共44个数
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(D, H_feat, W_feat)  # # [99,16,44] x[n][][m]=[0,...,255]一共16个数

        # D x H x W x 3  99,16,44,3 每个块里存储宽度上第几个像素，高度上第几个像素，深度的数值
        return torch.stack((x, y, d), -1)

    def create_frustum_2(self, depth_cfg, input_size, h):
        H_feat, W_feat = input_size
        d = torch.arange(*depth_cfg, dtype=torch.float).view(-1, 1, 1).expand(-1, H_feat,
                                                                              W_feat)  # [99,16,44],d[0]是1，d[98]全是99
        D = d.shape[0]
        x = torch.linspace(0, W_feat - 1, W_feat, dtype=torch.float).view(1, 1, W_feat).expand(D, H_feat, W_feat)  # [99,16,44] x[n][m]=[0,...,703]一共44个数
        y = torch.linspace(0, H_feat - 1, H_feat, dtype=torch.float).view(1, H_feat, 1).expand(D, H_feat, W_feat)  # # [99,16,44] x[n][][m]=[0,...,255]一共16个数

        # D x H x W x 3  99,16,44,3 每个块里存储宽度上第几个像素，高度上第几个像素，深度的数值
        return torch.stack((x, y, d), -1)

    def get_ground_depth(self, RT_1, K, post_rots, post_trans):
        start = time.perf_counter()  # %%%
        B, _, _ = RT_1.shape  #12,4,4
        # post-transformation，图像预处理逆变换，3里存的是每个格对应的真实图像2.5D坐标
        # B x N x D x H x W x 3
        frustum = self.create_frustum([1.0, 2, 1.0], (256, 704), 2)  # [99,16,44,3]这几个数啊，你要改config的话必须都改掉%%%,我改成1呢？
        # end_1 = time.perf_counter()  # %%%
        points = frustum.to(RT_1) - post_trans.view(B, 1, 1, 1, 3)  # 12 99 16 44 3,减去平移，frustum维度改变是因为广播机制
        img = torch.inverse(post_rots).view(B, 1, 1, 1, 3, 3) .matmul(points.unsqueeze(-1))  # 12 99 16 44 3 1
        # end_2 = time.perf_counter()  # %%%
        img2 = torch.squeeze(img, -1)
        img3 = img2[:, 0, :, :, :]  # 深度只取depth=0,最后得三维不要深度维
        img4 = img3[:, :, :, :-1]  # 先列，最大值1600,后行，最大值900
        # print("运行耗时1:", end_1 - start, "运行耗时2:", end_2 - start)
        return img4

    def project_3d_boxes_to_multiple_images(self, points_3D, Ks, RT_1s, bda):
        RTs = torch.inverse(RT_1s)
        image_width, image_height = int(Ks[0, 0, 2] * 2), int(Ks[0, 1, 2] * 2)
        B = Ks.shape[0]  # 图像数量
        N = points_3D.shape[0]  # 目标框的数量
        # 初始化一个图像批次
        images = torch.zeros((B, image_height, image_width), dtype=torch.uint8)
        if N!=0:  # 为了防止有时候没样本
            points_3D = points_3D.view(-1, 3)
            bda_1 = torch.inverse(bda)
            points_3D = bda_1 @ points_3D.T
            points_3D = points_3D.T.view(N, -1, 3)

            # 展平所有目标框的顶点
            flattened_points = points_3D.reshape(-1, 3)

            # 对每个图像进行处理
            for b in range(B):
                K = Ks[b]
                RT = RTs[b]

                # 投影三维点到二维图像平面
                ones = torch.ones((flattened_points.shape[0], 1))
                points_3D_hom = torch.cat((flattened_points, ones), dim=1)
                points_2D_hom = K @ RT[:3, :] @ points_3D_hom.T
                points_2D = points_2D_hom[:2, :] / points_2D_hom[2, :]
                points_2D = points_2D.T.view(N, 8, 2)

                # 计算所有目标框的外接矩形的最小和最大坐标
                min_coords, _ = torch.min(points_2D.view(N, -1, 2), dim=1)
                max_coords, _ = torch.max(points_2D.view(N, -1, 2), dim=1)

                # 确保坐标在图像范围内
                min_coords = torch.clamp(min_coords, 0)
                max_coords = torch.clamp(max_coords, max=torch.tensor([image_width - 1, image_height - 1]))

                # 绘制所有外接矩形到当前图像
                for min_coord, max_coord in zip(min_coords, max_coords):
                    min_x, min_y = min_coord.type(torch.int32)
                    max_x, max_y = max_coord.type(torch.int32)
                    images[b, min_y:max_y + 1, min_x:max_x + 1] = 1

        return images

    def sample_images_tensorized(self,object_images, sample):
        B, H, W = object_images.shape
        _, h, w, _ = sample.shape

        # 调整采样坐标并限制其范围
        sample = torch.clamp(sample - 1, min=0)

        # 生成网格坐标
        batch_idx = torch.arange(B)[:, None, None]

        # 使用网格坐标和采样坐标生成索引，同时将索引张量转换为 long 数据类型
        u = sample[..., 0].clamp(max=W - 1).long()
        v = sample[..., 1].clamp(max=H - 1).long()

        # 进行采样
        object_images_sample = object_images[batch_idx, v, u]

        # 生成越界掩码，并将越界的采样点设置为0
        mask = (sample[..., 0] >= H) | (sample[..., 1] >= W) | (sample[..., 0] < 0) | (sample[..., 1] < 0)
        object_images_sample[mask] = 0

        return object_images_sample


@PIPELINES.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
