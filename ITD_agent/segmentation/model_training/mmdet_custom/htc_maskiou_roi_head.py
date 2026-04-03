from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.models.roi_heads.htc_roi_head import HybridTaskCascadeRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList


@MODELS.register_module()
class HTCMaskScoringRoIHead(HybridTaskCascadeRoIHead):
    """HTC roi head with MaskIoU scoring on the final mask stage."""

    def __init__(self, mask_iou_head: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        if mask_iou_head is None:
            raise ValueError("mask_iou_head must be provided for HTCMaskScoringRoIHead")
        self.mask_iou_head = MODELS.build(mask_iou_head)

    def _mask_forward(
        self,
        stage: int,
        x: Tuple[Tensor],
        rois: Tensor,
        semantic_feat: Optional[Tensor] = None,
        training: bool = True,
    ) -> Dict[str, Tensor]:
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[: mask_roi_extractor.num_inputs], rois)

        if self.with_semantic and "mask" in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat], rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats = mask_feats + mask_semantic_feat

        if training:
            if self.mask_info_flow:
                last_feat = None
                for i in range(stage):
                    last_feat = self.mask_head[i](mask_feats, last_feat, return_logits=False)
                mask_preds = mask_head(mask_feats, last_feat, return_feat=False)
            else:
                mask_preds = mask_head(mask_feats, return_feat=False)
            return dict(mask_preds=mask_preds, mask_feats=mask_feats)

        aug_masks = []
        last_feat = None
        final_mask_preds = None
        for i in range(self.num_stages):
            current_mask_head = self.mask_head[i]
            if self.mask_info_flow:
                final_mask_preds, last_feat = current_mask_head(mask_feats, last_feat)
            else:
                final_mask_preds = current_mask_head(mask_feats)
        aug_masks.append(final_mask_preds)
        return dict(mask_preds=aug_masks, final_mask_preds=final_mask_preds, mask_feats=mask_feats)

    def mask_loss(
        self,
        stage: int,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        semantic_feat: Optional[Tensor] = None,
    ) -> dict:
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(
            stage=stage,
            x=x,
            rois=pos_rois,
            semantic_feat=semantic_feat,
            training=True,
        )

        mask_head = self.mask_head[stage]
        mask_loss_and_target = mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[stage],
        )
        mask_results.update(mask_loss_and_target)

        if stage != self.num_stages - 1 or mask_results.get("loss_mask") is None:
            return mask_results

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        pos_mask_pred = mask_results["mask_preds"][range(mask_results["mask_preds"].size(0)), pos_labels]
        mask_iou_pred = self.mask_iou_head(mask_results["mask_feats"], pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]
        loss_mask_iou = self.mask_iou_head.loss_and_target(
            pos_mask_iou_pred,
            pos_mask_pred,
            mask_results["mask_targets"],
            sampling_results,
            batch_gt_instances,
            self.train_cfg[stage],
        )
        mask_results["loss_mask"].update(loss_mask_iou)
        return mask_results

    def predict_mask(
        self,
        x: Tuple[Tensor],
        semantic_heat: Tensor,
        batch_img_metas: List[dict],
        results_list: InstanceList,
        rescale: bool = False,
    ) -> InstanceList:
        num_imgs = len(batch_img_metas)
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas=batch_img_metas,
                device=mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary,
            )
            return results_list

        num_mask_rois_per_img = [len(res) for res in results_list]
        labels = torch.cat([res.labels for res in results_list])
        mask_results = self._mask_forward(
            stage=-1,
            x=x,
            rois=mask_rois,
            semantic_feat=semantic_heat,
            training=False,
        )
        aug_masks = [
            [mask.sigmoid().detach() for mask in mask_preds.split(num_mask_rois_per_img, 0)]
            for mask_preds in mask_results["mask_preds"]
        ]

        merged_masks = []
        for i in range(num_imgs):
            aug_mask = [mask[i] for mask in aug_masks]
            merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
            merged_masks.append(merged_mask)

        results_list = self.mask_head[-1].predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True,
        )

        final_mask_preds = mask_results["final_mask_preds"]
        mask_iou_preds = self.mask_iou_head(
            mask_results["mask_feats"], final_mask_preds[range(labels.size(0)), labels]
        )
        mask_iou_preds = mask_iou_preds.split(num_mask_rois_per_img, 0)
        return self.mask_iou_head.predict_by_feat(mask_iou_preds=mask_iou_preds, results_list=results_list)
