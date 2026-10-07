import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
try:
    from torchsummary import summary
except ImportError:
    summary = None
import torchvision.models as models

class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False,
                 time_inject=False, num_frames=8, ref_idx=0,
                 use_external_time=False, time_scales=(1, 2)):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.time_inject = time_inject
        self.use_external_time = use_external_time
        self.num_frames = num_frames
        self.ref_idx = ref_idx
        self.num_other = num_frames - 1
        self.time_scales = time_scales

        if time_inject and not use_external_time:
            self.num_groups = num_frames + 1
            self.in_ch_per_group = 3
            self.out_ch_per_group = out_ch // self.num_groups
            self.frame_out_ch = self.out_ch_per_group * self.num_groups
            self.conv_frame = nn.Sequential(
                nn.Conv2d(self.in_ch_per_group, self.out_ch_per_group, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.out_ch_per_group, self.out_ch_per_group, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.out_ch_per_group, self.out_ch_per_group, 3, padding=1),
                nn.ReLU()
            )
            self.frame_project = (
                nn.Identity() if self.frame_out_ch == out_ch
                else nn.Conv2d(self.frame_out_ch, out_ch, kernel_size=1)
            )
            in_time = self.num_other * 3 * len(time_scales)
            time_hidden = max(in_time // 4, 8)
            self.time_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_time, time_hidden, 1),
                nn.ReLU(),
                nn.Conv2d(time_hidden, in_time, 1),
                nn.Sigmoid()
            )
            self.time_fuse = nn.Sequential(
                nn.Conv2d(in_time, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()
            )

        if use_external_time:
            in_time_external = self.num_other * 3
            self.external_time_fuse = nn.Sequential(
                nn.Conv2d(in_time_external, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1), nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1), nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid()
            )

    def forward(self, data, time_map=None):
        B, _, H, W = data.shape

        if self.time_inject and not self.use_external_time:
            data_reshaped = data.view(B, self.num_groups, self.in_ch_per_group, H, W)
            fm_list = []
            for g in range(self.num_groups):
                frame_feat = self.conv_frame(data_reshaped[:, g])
                fm_list.append(frame_feat)
            fm = torch.cat(fm_list, dim=1)
            fm = self.frame_project(fm)

            burst = data[:, :self.num_frames * 3].view(B, self.num_frames, 3, H, W)
            burst_gray = burst.mean(dim=2, keepdim=True)
            eps = 1e-6
            time_maps_all = []

            for scale in self.time_scales:
                pool_size = 2 * scale
                gray_pool = F.avg_pool2d(
                    burst_gray.view(B * self.num_frames, 1, H, W),
                    kernel_size=pool_size
                )
                _, _, H_l, W_l = gray_pool.shape
                ll = gray_pool.view(B, self.num_frames, 1, H_l, W_l)

                ref_ll = ll[:, self.ref_idx:self.ref_idx+1]
                other_idx = [i for i in range(self.num_frames) if i != self.ref_idx]
                other_ll = ll[:, other_idx]
                ref_expand = ref_ll.expand(-1, self.num_other, -1, -1, -1)

                map1 = (ref_expand + other_ll) / 2.0
                map2 = other_ll - ref_expand
                map3 = (ref_expand * other_ll) / (ref_expand**2 + other_ll**2 + eps)

                time_map = torch.cat([
                    map1.squeeze(2),
                    map2.squeeze(2),
                    map3.squeeze(2)
                ], dim=1)
                time_map_up = F.interpolate(time_map, size=(H, W), mode='bilinear', align_corners=False)
                time_maps_all.append(time_map_up)

            time_cat = torch.cat(time_maps_all, dim=1)
            time_cat = time_cat * self.time_att(time_cat)
            residual = self.time_fuse(time_cat)
            fm = fm + residual
        else:
            fm = self.conv1(data)

        if hasattr(self, 'external_time_fuse') and time_map is not None:
            residual = self.external_time_fuse(time_map)
            fm = fm + residual

        if self.channel_att:
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1,1)),
                                 F.adaptive_max_pool2d(fm, (1,1))], dim=1)
            fm = fm * self.att_c(fm_pool)
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True),
                                 torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            fm = fm * self.att_s(fm_pool)
        return fm

class KPN(nn.Module):
    def __init__(self, color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length

        self.conv1 = Basic(in_channel, 72, channel_att=False, spatial_att=False,
                           time_inject=True, num_frames=burst_length, ref_idx=0,
                           time_scales=(1,))
        self.conv1_proj = nn.Conv2d(72, 64, kernel_size=1, stride=1, padding=0)

        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False,
                           time_inject=True, use_external_time=True,
                           num_frames=burst_length, ref_idx=0)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False,
                           time_inject=True, use_external_time=True,
                           num_frames=burst_length, ref_idx=0)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False,
                           time_inject=True, use_external_time=True,
                           num_frames=burst_length, ref_idx=0)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    def compute_input_time_map(self, burst_gray, pool_scale, ref_idx=0):
        B, T, _, H, W = burst_gray.shape
        gray_pool = F.avg_pool2d(
            burst_gray.view(B * T, 1, H, W),
            kernel_size=pool_scale
        )
        _, _, H_l, W_l = gray_pool.shape
        ll = gray_pool.view(B, T, 1, H_l, W_l)

        ref_ll = ll[:, ref_idx:ref_idx + 1]
        other_idx = [i for i in range(T) if i != ref_idx]
        other_ll = ll[:, other_idx]
        eps = 1e-6
        ref_expand = ref_ll.expand(-1, T - 1, -1, -1, -1)

        map1 = (ref_expand + other_ll) / 2.0
        map2 = other_ll - ref_expand
        map3 = (ref_expand * other_ll) / (ref_expand**2 + other_ll**2 + eps)

        return torch.cat([
            map1.squeeze(2),
            map2.squeeze(2),
            map3.squeeze(2)
        ], dim=1)

    def forward(self, data_with_est, data, white_level=1.0):
        B, _, H, W = data_with_est.shape
        burst_raw = data_with_est[:, :self.burst_length * 3]
        burst_gray = burst_raw.view(B, self.burst_length, 3, H, W).mean(dim=2, keepdim=True)

        time_map_2 = self.compute_input_time_map(burst_gray, pool_scale=2, ref_idx=0)
        time_map_4 = self.compute_input_time_map(burst_gray, pool_scale=4, ref_idx=0)
        time_map_8 = self.compute_input_time_map(burst_gray, pool_scale=8, ref_idx=0)

        conv1 = self.conv1(data_with_est)
        conv1 = self.conv1_proj(conv1)

        conv2_input = F.avg_pool2d(conv1, kernel_size=2, stride=2)
        conv2 = self.conv2(conv2_input, time_map=time_map_2)
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2), time_map=time_map_4)
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2), time_map=time_map_8)
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))

        return self.kernel_pred(data, core, white_level)


class KernelConv(nn.Module):
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0):
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
                for i in range(K):
                    for j in range(K):
                        img_stack.append(frame_pad[..., i:i + height, j:j + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        pred_img_i = pred_img_i / white_level
        pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)
        return pred_img_i, pred_img


class LossFunc(nn.Module):
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)


class LossBasic(nn.Module):
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))


class LossAnneal(nn.Module):
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss


class TensorGradient(nn.Module):
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )


if __name__ == '__main__':
    kpn = KPN(6, 5*5*6, True, True).cuda()
    print(summary(kpn, (6, 224, 224), batch_size=4))
