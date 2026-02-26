# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from rfdetr.models.segmentation_head import point_sample
from rfdetr.util.box_ops import batch_dice_loss, batch_sigmoid_ce_loss, box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25, use_pos_only: bool = False,
                 use_position_modulated_cost: bool = False, mask_point_sample_ratio: int = 16, cost_mask_ce: float = 1, cost_mask_dice: float = 1,
                 matcher_chunk_size: int = 0):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice
        self.matcher_chunk_size = int(matcher_chunk_size)

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "masks": Tensor of dim [num_target_boxes, H, W] containing the target mask coordinates
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        if num_queries % group_detr != 0:
            raise ValueError(f"num_queries({num_queries}) must be divisible by group_detr({group_detr})")

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # Keep matcher numerically stable even if a bad sample introduces NaN/Inf.
        tgt_bbox = torch.nan_to_num(tgt_bbox, nan=0.0, posinf=1.0, neginf=0.0)
        sizes = [len(v["boxes"]) for v in targets]

        masks_present = "masks" in targets[0]
        tgt_masks = torch.cat([v["masks"] for v in targets]) if masks_present else None

        g_num_queries = num_queries // group_detr
        chunk_size = self.matcher_chunk_size if self.matcher_chunk_size > 0 else bs * g_num_queries

        alpha = 0.25
        gamma = 2.0
        indices = []

        for g_i in range(group_detr):
            start = g_i * g_num_queries
            end = start + g_num_queries

            flat_pred_logits = outputs["pred_logits"][:, start:end, :].flatten(0, 1)
            out_bbox = outputs["pred_boxes"][:, start:end, :].flatten(0, 1)
            flat_pred_logits = torch.nan_to_num(flat_pred_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            out_bbox = torch.nan_to_num(out_bbox, nan=0.0, posinf=1.0, neginf=0.0)

            pred_masks_logits = None
            tgt_masks_flat = None
            if masks_present:
                if isinstance(outputs["pred_masks"], torch.Tensor):
                    out_masks = outputs["pred_masks"][:, start:end, ...].flatten(0, 1)
                    num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio
                    point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
                    pred_masks_logits = point_sample(
                        out_masks.unsqueeze(1),
                        point_coords.repeat(out_masks.shape[0], 1, 1),
                        align_corners=False,
                    ).squeeze(1)
                else:
                    spatial_features = outputs["pred_masks"]["spatial_features"]
                    query_features = outputs["pred_masks"]["query_features"][:, start:end, :]
                    bias = outputs["pred_masks"]["bias"]

                    num_points = spatial_features.shape[-2] * spatial_features.shape[-1] // self.mask_point_sample_ratio
                    point_coords = torch.rand(1, num_points, 2, device=spatial_features.device)
                    sampled = point_sample(
                        spatial_features,
                        point_coords.repeat(spatial_features.shape[0], 1, 1),
                        align_corners=False,
                    )
                    pred_masks_logits = torch.einsum("bcp,bnc->bnp", sampled, query_features) + bias
                    pred_masks_logits = pred_masks_logits.flatten(0, 1)
                pred_masks_logits = torch.nan_to_num(pred_masks_logits, nan=0.0, posinf=20.0, neginf=-20.0)

                assert tgt_masks is not None
                tgt_masks_cast = tgt_masks.to(pred_masks_logits.dtype)
                tgt_masks_flat = point_sample(
                    tgt_masks_cast.unsqueeze(1),
                    point_coords.repeat(tgt_masks_cast.shape[0], 1, 1),
                    align_corners=False,
                    mode="nearest",
                ).squeeze(1)

            cost_chunks = []
            for row_start in range(0, flat_pred_logits.shape[0], chunk_size):
                row_end = min(row_start + chunk_size, flat_pred_logits.shape[0])
                pred_logits_chunk = flat_pred_logits[row_start:row_end]
                out_bbox_chunk = out_bbox[row_start:row_end]

                out_prob = pred_logits_chunk.sigmoid()
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-F.logsigmoid(-pred_logits_chunk))
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(pred_logits_chunk))
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

                cost_bbox = torch.cdist(out_bbox_chunk, tgt_bbox, p=1)
                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox_chunk),
                    box_cxcywh_to_xyxy(tgt_bbox),
                )
                cost_giou = -giou

                C_chunk = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                if masks_present and pred_masks_logits is not None and tgt_masks_flat is not None:
                    pred_masks_chunk = pred_masks_logits[row_start:row_end]
                    cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_chunk, tgt_masks_flat)
                    cost_mask_dice = batch_dice_loss(pred_masks_chunk, tgt_masks_flat)
                    C_chunk = C_chunk + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice

                cost_chunks.append(C_chunk)

            C_g = torch.cat(cost_chunks, dim=0).view(bs, g_num_queries, -1).float().cpu()
            finite_mask = torch.isfinite(C_g)
            if not finite_mask.all():
                if finite_mask.any():
                    finite_max = C_g[finite_mask].max()
                    fill_value = (finite_max + 1.0).item()
                else:
                    fill_value = 1e6
                C_g = torch.nan_to_num(C_g, nan=fill_value, posinf=fill_value, neginf=fill_value)

            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + start]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, indices_g)
                ]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    matcher_chunk_size = int(getattr(args, "matcher_chunk_size", 0) or 0)
    if args.segmentation_head:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
            matcher_chunk_size=matcher_chunk_size,)
    else:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            matcher_chunk_size=matcher_chunk_size,
        )
