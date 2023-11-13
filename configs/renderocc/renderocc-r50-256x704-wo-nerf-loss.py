'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-11-01 09:10:57
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
_base_ = ['./bevstereo-occ-r50-256x704.py']

model = dict(
    type='RenderOcc',
    use_3d_loss=True,
    final_softplus=True,
    nerf_head=None
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)


depth_gt_path = './data/nuscenes/depth_gt'
semantic_gt_path = './data/nuscenes/seg_gt_lidarseg'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_train-quarter.pkl',
        use_rays=False,
        depth_gt_path=depth_gt_path,
        semantic_gt_path=semantic_gt_path,
        aux_frames=[-3,-2,-1,1,2,3],
        max_ray_nums=38400,
    )
)


runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(
    interval=50,
)
