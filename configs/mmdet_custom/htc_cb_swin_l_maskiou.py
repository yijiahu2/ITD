_base_ = "/home/xth/mmdetection331/configs/htc/htc-without-semantic_r50_fpn_1x_coco.py"

custom_imports = dict(
    imports=["ITD_agent.segmentation.model_training.mmdet_custom"],
    allow_failed_imports=False,
)

swin_backbone_cfg = dict(
    type="SwinTransformer",
    pretrain_img_size=384,
    embed_dims=192,
    patch_size=4,
    window_size=12,
    mlp_ratio=4,
    depths=[2, 2, 18, 2],
    num_heads=[6, 12, 24, 48],
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.3,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    with_cp=False,
    convert_weights=True,
    init_cfg=None,
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="CBSwinTransformer",
        num_backbones=2,
        out_indices=(0, 1, 2, 3),
        backbone_cfg=swin_backbone_cfg,
    ),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    roi_head=dict(
        type="HTCMaskScoringRoIHead",
        mask_iou_head=dict(
            type="MaskIoUHead",
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=80,
        ),
    ),
)
