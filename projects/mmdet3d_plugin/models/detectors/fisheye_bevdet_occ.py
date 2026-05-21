# Copyright (c) OpenMMLab. All rights reserved.
# Fisheye BEVDepth OCC detector — handles 8-item img_inputs (with distortions)

import torch
import torch.nn.functional as F

from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
from .bevdet_occ import BEVDepthOCC


@DETECTORS.register_module()
class FisheyeBEVDepthOCC(BEVDepthOCC):
    """BEVDepth OCC detector adapted for fisheye cameras.

    Overrides prepare_inputs and extract_img_feat to handle the 8th
    element in img_inputs: distortion coefficients (B, N, 4).
    """

    def prepare_inputs(self, inputs):
        assert len(inputs) == 8
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda, distortions = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
        global2keyego = torch.linalg.inv(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda, distortions]

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, distortions = \
            self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(imgs)
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)

        x, depth = self.img_view_transformer(
            [x, sensor2keyegos, ego2globals, intrins, post_rots,
             post_trans, bda, distortions, mlp_input])
        x = self.bev_encoder(x)
        return [x], depth

    def simple_test_occ(self, img_feats, img_metas=None):
        """Use get_occ (CPU argmax) instead of get_occ_gpu (GPU)."""
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ(outs, img_metas)
        return occ_preds
