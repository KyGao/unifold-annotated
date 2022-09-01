import os
os.chdir("/root/Uni-Fold/unifold")
os.getcwd()

import math
import torch
import torch.nn as nn
from typing import Tuple

from unifold.modules.common import Linear, SimpleModuleList
from unifold.modules.attentions import gen_attn_mask
from unifold.data.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from unifold.modules.frame import Rotation, Frame, Quaternion

from unicore.utils import (
    one_hot,
    dict_multimap,
    permute_final_dims,
)
from unicore.modules import LayerNorm, softmax_dropout


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def torsion_angles_to_frames(
    frame: Frame, # [10, 20]
    alpha: torch.Tensor, # [10, 20, 7, 2]
    aatype: torch.Tensor, # [10, 20]
    default_frames: torch.Tensor, # [21, 8, 4, 4], one [8, 4, 4] for each AA
):
    default_frame = Frame.from_tensor_4x4(default_frames[aatype, ...]) # [10, 20, 8]  read default frame for each AA type

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2)) # [1, 1, 1, 2] 
    bb_rot[..., 1] = 1

    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2) # [10, 20, 8, 2]

    # [1,  0,   0  ]
    # [0, sin, -cos]
    # [0, cos, sin ]
    all_rots = alpha.new_zeros(default_frame.get_rots().rot_mat.shape) # [10, 20, 8, 3, 3]
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Frame(Rotation(mat=all_rots), None) # [10, 20, 8]

    all_frames = default_frame.compose(all_rots) # original frame stack predicted rotations

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Frame.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    # 0-7 group frame rotations (no translation needed)
    # 0: bb + eye + eye
    # 1: bb + eye + pred
    # 2, 3, 4: bb + default_frame + pred
    # 5, 6, 7: bb + default_frame + pred + ...
    all_frames_to_global = frame[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    frame: Frame, # [10, 20, 8]
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    group_mask = group_idx[aatype, ...] # [10, 20, 14], eg: tensor([0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), tensor([0, 0, 0, 3, 0, 4, 5, 6, 7, 7, 7, 0, 0, 0]), tensor([0, 0, 0, 3, 0, 4, 5, 5, 0, 0, 0, 0, 0, 0]), tensor([0, 0, 0, 3, 0, 4, 5, 5, 0, 0, 0, 0, 0, 0]), tensor([0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0])
    group_mask = one_hot( # [10, 20, 14, 8]
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    t_atoms_to_global = frame[..., None, :] * group_mask # [10, 20, 14, 8], for the last dim, only according group (1 of 8) has value
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1)) # [10, 20, 14]

    atom_mask = atom_mask[aatype, ...].unsqueeze(-1) # [10, 20, 14, 1]

    lit_positions = lit_positions[aatype, ...] # [10, 20, 14, 3]
    pred_positions = t_atoms_to_global.apply(lit_positions) # [10, 20, 14, 3]
    pred_positions = pred_positions * atom_mask

    return pred_positions # [10, 20, 14, 8]


class SideChainAngleResnetIteration(nn.Module):
    def __init__(self, d_hid):
        super(SideChainAngleResnetIteration, self).__init__()

        self.d_hid = d_hid

        self.linear_1 = Linear(self.d_hid, self.d_hid, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.d_hid, self.d_hid, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:

        x = self.act(s)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        return x + s


class SidechainAngleResnet(nn.Module):
    def __init__(self, d_in, d_hid, num_blocks, num_angles):
        super(SidechainAngleResnet, self).__init__()

        self.linear_in = Linear(d_in, d_hid)
        self.act = nn.GELU()
        self.linear_initial = Linear(d_in, d_hid)

        self.layers = SimpleModuleList()
        for _ in range(num_blocks):
            self.layers.append(SideChainAngleResnetIteration(d_hid=d_hid))

        self.linear_out = Linear(d_hid, num_angles * 2)

    def forward(
        self, s: torch.Tensor, initial_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        initial_s = self.linear_initial(self.act(initial_s))
        s = self.linear_in(self.act(s))

        s = s + initial_s

        for layer in self.layers:
            s = layer(s)

        s = self.linear_out(self.act(s)) # [10, 20, 14]

        s = s.view(s.shape[:-1] + (-1, 2)) # [10, 20, 7, 2]

        unnormalized_s = s
        norm_denom = torch.sqrt( # L2 norm
            torch.clamp(
                torch.sum(s.float() ** 2, dim=-1, keepdim=True),
                min=1e-12,
            )
        )
        s = s.float() / norm_denom

        return unnormalized_s, s.type(unnormalized_s.dtype)


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hid: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        separate_kv: bool = False,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        super(InvariantPointAttention, self).__init__()

        self.d_hid = d_hid
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.eps = eps

        hc = self.d_hid * self.num_heads
        self.linear_q = Linear(d_single, hc, bias=bias)
        self.separate_kv = separate_kv
        if self.separate_kv:
            self.linear_k = Linear(d_single, hc, bias=bias)
            self.linear_v = Linear(d_single, hc, bias=bias)
        else:
            self.linear_kv = Linear(d_single, 2 * hc, bias=bias)

        hpq = self.num_heads * self.num_qk_points * 3
        self.linear_q_points = Linear(d_single, hpq)
        hpk = self.num_heads * self.num_qk_points * 3
        hpv = self.num_heads * self.num_v_points * 3
        if self.separate_kv:
            self.linear_k_points = Linear(d_single, hpk)
            self.linear_v_points = Linear(d_single, hpv)
        else:
            hpkv = hpk + hpv
            self.linear_kv_points = Linear(d_single, hpkv)

        self.linear_b = Linear(d_pair, self.num_heads)

        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.num_heads * (d_pair + self.d_hid + self.num_v_points * 4)
        self.linear_out = Linear(concat_out_dim, d_single, init="final")

        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,  # [10, 20, 384]
        z: torch.Tensor,  # [10, 20, 20, 128]
        f: Frame,
        square_mask: torch.Tensor, # [10, 1, 20, 20]
    ) -> torch.Tensor:
        q = self.linear_q(s) # [10, 20, 192]

        q = q.view(q.shape[:-1] + (self.num_heads, -1)) # [10, 20, 12, 16] (head is 12)

        if self.separate_kv:
            k = self.linear_k(s)
            v = self.linear_v(s)
            k = k.view(k.shape[:-1] + (self.num_heads, -1))
            v = v.view(v.shape[:-1] + (self.num_heads, -1))
        else:
            kv = self.linear_kv(s) # [10, 20, 384]
            kv = kv.view(kv.shape[:-1] + (self.num_heads, -1)) # [10, 20, 12, 32]
            k, v = torch.split(kv, self.d_hid, dim=-1) # [10, 20, 12, 16] * 2, self.d_hid = 16

        q_pts = self.linear_q_points(s) # [10, 20, 384] => [10, 20, 144], 144=12*4*3

        def process_points(pts, no_points):
            shape = pts.shape[:-1] + (pts.shape[-1] // 3, 3) # [10, 20, 48, 3], 3 because 3D
            if self.separate_kv:
                # alphafold-multimer uses different layout
                pts = pts.view(pts.shape[:-1] + (self.num_heads, no_points * 3))
            pts = torch.split(pts, pts.shape[-1] // 3, dim=-1) # [10, 20, 48] * 3
            pts = torch.stack(pts, dim=-1).view(*shape) # [10, 20, 48, 3]
            # [10, 20, 48, 3]      rotate and translate on 10 * 20 points * 48 virtual pts (12 head, 4 n query)
            pts = f[..., None].apply(pts) 

            pts = pts.view(pts.shape[:-2] + (self.num_heads, no_points, 3)) # [10, 20, 12, 4, 3]
            return pts # [10, 20, 12, 4, 3]

        # [10, 20, 144] => [10, 20, 12, 4, 3],  split and transform each point
        q_pts = process_points(q_pts, self.num_qk_points) 

        if self.separate_kv:
            k_pts = self.linear_k_points(s)
            v_pts = self.linear_v_points(s)
            k_pts = process_points(k_pts, self.num_qk_points)
            v_pts = process_points(v_pts, self.num_v_points)
        else:
            kv_pts = self.linear_kv_points(s) # [10, 20, 384] => [10, 20, 432]

            kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
            kv_pts = torch.stack(kv_pts, dim=-1) # [10, 20, 144, 3]
            kv_pts = f[..., None].apply(kv_pts)  # [10, 20, 36, 4, 3]

            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.num_heads, -1, 3)) # [10, 20, 12, 12, 3]

            k_pts, v_pts = torch.split(
                kv_pts, [self.num_qk_points, self.num_v_points], dim=-2
            ) # [10, 20, 12, 4, 3], [10, 20, 12, 8, 3]

        bias = self.linear_b(z) # [10, 20, 20, 128] => [10, 20, 20, 12], one scalar for each head

        attn = torch.matmul( # [10, 12, 20, 20]
            permute_final_dims(q, (1, 0, 2)),
            permute_final_dims(k, (1, 2, 0)),
        )
        attn = attn * math.sqrt(1.0 / (3 * self.d_hid))  # wL and c
        attn = attn + (math.sqrt(1.0 / 3) * permute_final_dims(bias, (2, 0, 1))) # bias

        # [10, 20, 1, 12, 4, 3] - [10, 1, 20, 12, 4, 3] = [10, 20, 20, 12, 4, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5) 
        pt_att = pt_att.float() ** 2

        pt_att = pt_att.sum(dim=-1) # [10, 20, 20, 12, 4]
        # [12], each head has one learnable weight, softplus to make it positive
        head_weights = self.softplus(self.head_weights).view( 
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.num_qk_points * 9.0 / 2))
        )
        pt_att *= head_weights * (-0.5) 

        pt_att = torch.sum(pt_att, dim=-1) # [10, 20, 20, 12, 4] => [10, 20, 20, 12]

        pt_att = permute_final_dims(pt_att, (2, 0, 1)) # [10, 12, 20, 20]
        attn += square_mask
        attn = softmax_dropout(attn, 0, self.training, bias=pt_att.type(attn.dtype)) # [10, 12, 20, 20]

        # [10, 12, 20, 20] * [10, 20, 12, 16] => [10, 20, 12, 16]
        o = torch.matmul(attn, v.transpose(-2, -3)).transpose(-2, -3) 
        o = o.contiguous().view(*o.shape[:-2], -1) # [10, 20, 192], combine all heads

        o_pts = torch.sum(
            (
                attn[..., None, :, :, None] # [10, 12, 1, 20, 20, 1]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :] # [10, 12, 3, 1, 20, 8]
            ),
            dim=-2,
        ) # [10, 12, 3, 20, 8]

        o_pts = permute_final_dims(o_pts, (2, 0, 3, 1)) # [10, 20, 12, 8, 3]
        o_pts = f[..., None, None].invert_apply(o_pts) # [10, 20, 12, 8, 3]

        o_pts_norm = torch.sqrt(torch.sum(o_pts.float() ** 2, dim=-1) + self.eps).type(
            o_pts.dtype
        ) # [10, 20, 12, 8], L1 norm of each point (each head and each point produce a scalar)

        o_pts_norm = o_pts_norm.view(*o_pts_norm.shape[:-2], -1) # [10, 20, 96]

        o_pts = o_pts.view(*o_pts.shape[:-3], -1, 3) # [10, 20, 12, 8, 3] => [10, 20, 96, 3]

        o_pair = torch.matmul(attn.transpose(-2, -3), z) # [10, 20, 12, 128]

        o_pair = o_pair.view(*o_pair.shape[:-2], -1) # [10, 20, 12, 1536]

        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pts, dim=-1), o_pts_norm, o_pair), dim=-1) # [10, 20, 2112]
        ) # [10, 20, 384]

        return s


class BackboneUpdate(nn.Module):
    def __init__(self, d_single):
        super(BackboneUpdate, self).__init__()
        self.linear = Linear(d_single, 6, init="final")

    def forward(self, s: torch.Tensor):
        return self.linear(s)


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.linear_1 = Linear(c, c, init="relu")
        self.linear_2 = Linear(c, c, init="relu")
        self.act = nn.GELU()
        self.linear_3 = Linear(c, c, init="final")

    def forward(self, s):
        s_old = s
        s = self.linear_1(s)
        s = self.act(s)
        s = self.linear_2(s)
        s = self.act(s)
        s = self.linear_3(s)

        s = s + s_old

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = SimpleModuleList()
        for _ in range(self.num_layers):
            self.layers.append(StructureModuleTransitionLayer(c))

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(c)

    def forward(self, s):
        for layer in self.layers:
            s = layer(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):
    def __init__(
        self,
        d_single,
        d_pair,
        d_ipa,
        d_angle,
        num_heads_ipa,
        num_qk_points,
        num_v_points,
        dropout_rate,
        num_blocks,
        no_transition_layers,
        num_resnet_blocks,
        num_angles,
        trans_scale_factor,
        separate_kv,
        ipa_bias,
        epsilon,
        inf,
        **kwargs,
    ):
        super(StructureModule, self).__init__()

        self.num_blocks = num_blocks
        self.trans_scale_factor = trans_scale_factor
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None
        self.inf = inf

        self.layer_norm_s = LayerNorm(d_single)
        self.layer_norm_z = LayerNorm(d_pair)

        self.linear_in = Linear(d_single, d_single)

        self.ipa = InvariantPointAttention(
            d_single,
            d_pair,
            d_ipa,
            num_heads_ipa,
            num_qk_points,
            num_v_points,
            separate_kv=separate_kv,
            bias=ipa_bias,
            eps=epsilon,
        )

        self.ipa_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_ipa = LayerNorm(d_single)

        self.transition = StructureModuleTransition(
            d_single,
            no_transition_layers,
            dropout_rate,
        )

        self.bb_update = BackboneUpdate(d_single)

        self.angle_resnet = SidechainAngleResnet(
            d_single,
            d_angle,
            num_resnet_blocks,
            num_angles,
        )

    def forward(
        self,
        s,
        z,
        aatype,
        mask=None,
    ):
        if mask is None:
            mask = s.new_ones(s.shape[:-1]) # [10, 20]

        # generate square mask
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = gen_attn_mask(square_mask, -self.inf).unsqueeze(-3) # [10, 1, 20, 20]
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        initial_s = s
        s = self.linear_in(s) # [10, 20, 384]

        # quat_encoder: Quaternion
        # backb_to_global: Frame
        quat_encoder = Quaternion.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            requires_grad=False,
        )
        backb_to_global = Frame(
            Rotation(
                mat=quat_encoder.get_rot_mats(), # diagonol matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            ), 
            quat_encoder.get_trans(), # [10, 20, 3], all zeros
        )
        outputs = []
        for i in range(self.num_blocks):

            s = s + self.ipa(s, z, backb_to_global, square_mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s) # FFN, still [10, 20, 384]

            # update quaternion encoder
            # use backb_to_global to avoid quat-to-rot conversion
            # after bb_update: [10, 20, 6], 3 for quaternion, 3 for translation
            quat_encoder = quat_encoder.compose_update_vec(
                self.bb_update(s), pre_rot_mat=backb_to_global.get_rots() 
            ) # quat_encoder.get_quats().shape = [10, 20, 4]

            # initial_s is always used to update the backbone
            unnormalized_angles, angles = self.angle_resnet(s, initial_s) # [10, 20, 7, 2] * 2

            # convert quaternion to rotation matrix
            backb_to_global = Frame(
                Rotation(
                    mat=quat_encoder.get_rot_mats(), # first time: identity matrix
                ),
                quat_encoder.get_trans(),
            )
            if i == self.num_blocks - 1: # all output: bb Frame and 7 angles
                all_frames_to_global = self.torsion_angles_to_frames( # [10, 20, 8] frame
                    backb_to_global.scale_translation(self.trans_scale_factor),
                    angles, # [10, 20, 7, 2]
                    aatype, # [10, 20]
                )

                pred_positions = self.frames_and_literature_positions_to_atom14_pos( # [10, 20, 14, 3]
                    all_frames_to_global, # [10, 20, 8]
                    aatype, # [10, 20]
                )

            preds = {
                "frames": backb_to_global.scale_translation(
                    self.trans_scale_factor # 10
                ).to_tensor_4x4(), # [10, 20, 4, 4]
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
            }

            outputs.append(preds)
            if i < (self.num_blocks - 1):
                # stop gradient in iteration, only for rotation?
                quat_encoder = quat_encoder.stop_rot_gradient()
                backb_to_global = backb_to_global.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["sidechain_frames"] = all_frames_to_global.to_tensor_4x4()
        outputs["positions"] = pred_positions
        outputs["single"] = s

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frames is None: # [21, 8, 4, 4]
            self.default_frames = torch.tensor( 
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None: # [21, 14]
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None: # [21, 14]
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None: # [21, 14, 3]
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def torsion_angles_to_frames(self, frame, alpha, aatype):
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(frame, alpha, aatype, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, frame, aatype):
        self._init_residue_constants(frame.get_rots().dtype, frame.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            frame,
            aatype,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )



if __name__ == '__main__':
    struct = StructureModule(        
        d_single=384,
        d_pair=128,
        d_ipa=16,
        d_angle=128,
        num_heads_ipa=12,
        num_qk_points=4,
        num_v_points=8,
        dropout_rate=0.1,
        num_blocks=8,
        no_transition_layers=1,
        num_resnet_blocks=2,
        num_angles=7,
        trans_scale_factor=10,
        separate_kv=False,
        ipa_bias=True,
        epsilon=1e-12,
        inf=1e5,
    )

    s = torch.randn((10, 20, 384))
    z = torch.randn((10, 20, 20, 128))
    aatype = [[i for i in range(20)] for j in range(10)] # [10, 20]
    struct(s, z, aatype)