from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


# pre, post processing modules
def softmax_aggregate(predicted_seg, object_ids):
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in predicted_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logits = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7)
              for n, seg in [(-1, bg_seg)] + list(predicted_seg.items())}
    logits_sum = torch.cat(list(logits.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logits[n] / logits_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    final_seg_wrongids = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    assert final_seg_wrongids.dtype == torch.int64
    final_seg = torch.zeros_like(final_seg_wrongids)
    for idx, obj_idx in enumerate(object_ids):
        final_seg[final_seg_wrongids == (idx + 1)] = obj_idx
    return final_seg, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in enumerate(object_ids)}


def get_required_padding(height, width, div):
    height_pad = (div - height % div) % div
    width_pad = (div - width % div) % div
    padding = [(width_pad + 1) // 2, width_pad // 2, (height_pad + 1) // 2, height_pad // 2]
    return padding


def apply_padding(x, y, padding):
    B, L, C, H, W = x.size()
    x = x.view(B * L, C, H, W)
    x = F.pad(x, padding, mode='reflect')
    _, _, height, width = x.size()
    x = x.view(B, L, C, height, width)
    y = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in y]
    return x, y


def unpad(tensor, padding):
    if isinstance(tensor, (dict, OrderedDict)):
        return {key: unpad(val, padding) for key, val in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        return [unpad(elem, padding) for elem in tensor]
    else:
        _, _, _, height, width = tensor.size()
        tensor = tensor[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]
        return tensor


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DeConv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('deconv', nn.ConvTranspose2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x


# encoding module
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv.models.densenet121(pretrained=True).features
        self.conv0 = backbone.conv0
        self.norm0 = backbone.norm0
        self.relu0 = backbone.relu0
        self.pool0 = backbone.pool0
        self.denseblock1 = backbone.denseblock1
        self.transition1 = backbone.transition1
        self.denseblock2 = backbone.denseblock2
        self.transition2 = backbone.transition2
        self.denseblock3 = backbone.denseblock3
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        s4 = x
        x = self.transition1(x)
        x = self.denseblock2(x)
        s8 = x
        x = self.transition2(x)
        x = self.denseblock3(x)
        s16 = x
        return {'s16': s16, 's8': s8, 's4': s4}


# matching module
class Matcher(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = Conv(in_c, out_c, 1, 1, 0)
        self.short = nn.Parameter(torch.Tensor([1, 1]))
        self.long = nn.Parameter(torch.Tensor([-1, -1]))

    def get_norm_key(self, x):
        key = self.conv(x)
        norm_key = key / key.norm(dim=1, keepdim=True)
        return norm_key

    def build_fine_template(self, norm_key, init_seg_16):
        B, _, H, W = norm_key.size()
        fine_key = norm_key.view(B, -1, H * W).transpose(1, 2)
        fine_key_bg = fine_key * init_seg_16[:, 0].view(B, H * W, 1)
        fine_key_fg = fine_key * init_seg_16[:, 1].view(B, H * W, 1)
        return [fine_key_bg, fine_key_fg]

    def build_coarse_template(self, fine_key):
        coarse_key_bg = fine_key[0].sum(1, keepdim=True).clamp(min=1e-7)
        coarse_key_fg = fine_key[1].sum(1, keepdim=True).clamp(min=1e-7)
        coarse_key_bg = coarse_key_bg / coarse_key_bg.norm(dim=2, keepdim=True)
        coarse_key_fg = coarse_key_fg / coarse_key_fg.norm(dim=2, keepdim=True)
        return [coarse_key_bg, coarse_key_fg]

    def update_state(self, norm_key, update_seg_16, state):
        B, C, H, W = norm_key.size()

        # update local matching template
        fine_key = norm_key.view(B, -1, H * W).transpose(1, 2)
        fine_key_bg = fine_key * update_seg_16[:, 0, :, :].view(B, H * W, 1)
        fine_key_fg = fine_key * update_seg_16[:, 1, :, :].view(B, H * W, 1)
        state['local'] = [fine_key_bg, fine_key_fg]

        # update seg sum and calculate update dynamic_inertia value
        state['prev_seg_16'] = update_seg_16
        seg_sum = torch.sum(update_seg_16, dim=[2, 3]).unsqueeze(2)
        state['seg_sum'] = state['seg_sum'] + seg_sum
        dynamic_inertia = 1 - seg_sum / state['seg_sum']

        # update coarse matching templates
        coarse_key_bg = fine_key_bg.sum(1, keepdim=True).clamp(min=1e-7)
        coarse_key_fg = fine_key_fg.sum(1, keepdim=True).clamp(min=1e-7)
        coarse_key_bg = coarse_key_bg / coarse_key_bg.norm(dim=2, keepdim=True)
        coarse_key_fg = coarse_key_fg / coarse_key_fg.norm(dim=2, keepdim=True)
        new_coarse_key_bg = dynamic_inertia[:, 0:1] * state['overall'][0] + (1 - dynamic_inertia[:, 0:1]) * coarse_key_bg
        new_coarse_key_fg = dynamic_inertia[:, 1:2] * state['overall'][1] + (1 - dynamic_inertia[:, 1:2]) * coarse_key_fg
        new_coarse_key_bg = new_coarse_key_bg / new_coarse_key_bg.norm(dim=2, keepdim=True)
        new_coarse_key_fg = new_coarse_key_fg / new_coarse_key_fg.norm(dim=2, keepdim=True)
        state['overall'] = [new_coarse_key_bg, new_coarse_key_fg]

        short_inertia = 1 / (1 + torch.exp(self.short))
        new_coarse_key_bg = short_inertia[0] * state['short-term'][0] + (1 - short_inertia[0]) * coarse_key_bg
        new_coarse_key_fg = short_inertia[1] * state['short-term'][1] + (1 - short_inertia[1]) * coarse_key_fg
        new_coarse_key_bg = new_coarse_key_bg / new_coarse_key_bg.norm(dim=2, keepdim=True)
        new_coarse_key_fg = new_coarse_key_fg / new_coarse_key_fg.norm(dim=2, keepdim=True)
        state['short-term'] = [new_coarse_key_bg, new_coarse_key_fg]

        long_inertia = 1 / (1 + torch.exp(self.long))
        new_coarse_key_bg = long_inertia[0] * state['long-term'][0] + (1 - long_inertia[0]) * coarse_key_bg
        new_coarse_key_fg = long_inertia[1] * state['long-term'][1] + (1 - long_inertia[1]) * coarse_key_fg
        new_coarse_key_bg = new_coarse_key_bg / new_coarse_key_bg.norm(dim=2, keepdim=True)
        new_coarse_key_fg = new_coarse_key_fg / new_coarse_key_fg.norm(dim=2, keepdim=True)
        state['long-term'] = [new_coarse_key_bg, new_coarse_key_fg]
        return state

    def forward(self, norm_key, state, sds):
        B, _, H, W = norm_key.size()

        # global matching
        score = torch.bmm(state['global'][0], norm_key.view(B, -1, H * W)).view(B, H * W, H, W)
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = torch.bmm(state['global'][1], norm_key.view(B, -1, H * W)).view(B, H * W, H, W)
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        global_score = torch.cat([bg_score, fg_score], dim=1)

        # local matching
        score = torch.bmm(state['local'][0], norm_key.view(B, -1, H * W)).view(B, H * W, H, W) * sds
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = torch.bmm(state['local'][1], norm_key.view(B, -1, H * W)).view(B, H * W, H, W) * sds
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        local_score = torch.cat([bg_score, fg_score], dim=1)

        # overall matching
        bg_score = torch.bmm(state['overall'][0], norm_key.view(B, -1, H * W)).view(B, 1, H, W)
        fg_score = torch.bmm(state['overall'][1], norm_key.view(B, -1, H * W)).view(B, 1, H, W)
        overall_score = torch.cat([bg_score, fg_score], dim=1)

        # short-term matching
        bg_score = torch.bmm(state['short-term'][0], norm_key.view(B, -1, H * W)).view(B, 1, H, W)
        fg_score = torch.bmm(state['short-term'][1], norm_key.view(B, -1, H * W)).view(B, 1, H, W)
        short_term_score = torch.cat([bg_score, fg_score], dim=1)

        # long-term matching
        bg_score = torch.bmm(state['long-term'][0], norm_key.view(B, -1, H * W)).view(B, 1, H, W)
        fg_score = torch.bmm(state['long-term'][1], norm_key.view(B, -1, H * W)).view(B, 1, H, W)
        long_term_score = torch.cat([bg_score, fg_score], dim=1)
        return torch.cat([global_score, local_score, overall_score, short_term_score, long_term_score], dim=1)


# decoding module
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvRelu(1024, 256, 1, 1, 0)
        self.blend1 = ConvRelu(256 + 10 + 2, 256, 3, 1, 1)
        self.cbam1 = CBAM(256)
        self.deconv1 = DeConv(256, 2, 4, 2, 1)
        self.conv2 = ConvRelu(512, 256, 1, 1, 0)
        self.blend2 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.cbam2 = CBAM(256)
        self.deconv2 = DeConv(256, 2, 4, 2, 1)
        self.conv3 = ConvRelu(256, 256, 1, 1, 0)
        self.blend3 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.cbam3 = CBAM(256)
        self.predictor = DeConv(256, 2, 6, 4, 1)

    def forward(self, feats, simscore, prev_seg_16):
        x = torch.cat([self.conv1(feats['s16']), simscore, prev_seg_16], dim=1)
        x = self.blend1(x)
        x = self.cbam1(x)
        s8 = self.deconv1(x)
        x = torch.cat([self.conv2(feats['s8']), s8], dim=1)
        x = self.blend2(x)
        x = self.cbam2(x)
        s4 = self.deconv2(x)
        x = torch.cat([self.conv3(feats['s4']), s4], dim=1)
        x = self.blend3(x)
        x = self.cbam3(x)
        x = self.predictor(x)
        return x


# VOS model
class VOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.matcher = Matcher(1024, 512)
        self.decoder = Decoder()

    def get_init_state(self, norm_key, init_seg_16):
        state = {}
        state['global'] = self.matcher.build_fine_template(norm_key, init_seg_16)
        state['local'] = state['global'].copy()
        state['overall'] = self.matcher.build_coarse_template(state['global'].copy())
        state['short-term'] = state['overall'].copy()
        state['long-term'] = state['overall'].copy()
        state['prev_seg_16'] = init_seg_16
        state['seg_sum'] = torch.sum(init_seg_16, dim=[2, 3]).unsqueeze(2)
        return state

    def update(self, norm_key, update_seg_16, full_state, object_ids):
        for k in object_ids:
            full_state[k] = self.matcher.update_state(norm_key, update_seg_16[k], full_state[k])
        return full_state

    def forward(self, feats, norm_key, full_state, object_ids, sds):
        segscore = {}
        for k in object_ids:
            simscore = self.matcher(norm_key, full_state[k], sds)
            segscore[k] = self.decoder(feats, simscore, full_state[k]['prev_seg_16'])
        return segscore


# TBD model
class TBD(nn.Module):
    def __init__(self):
        super().__init__()
        self.vos = VOS()
        self.dsf = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())

    def forward(self, x, given_masks=None, dist=None, val_frame_ids=None):
        B, L, _, H0, W0 = x.size()

        # initialize VOS
        required_padding = get_required_padding(H0, W0, 16)
        if tuple(required_padding) != (0, 0, 0, 0):
            x, given_masks = apply_padding(x, given_masks, required_padding)
        _, _, _, H0, W0 = x.size()
        video_frames = [elem.view(B, 3, H0, W0) for elem in x.split(1, dim=1)]
        feats = self.vos.encoder(video_frames[0])
        norm_key = self.vos.matcher.get_norm_key(feats['s16'])
        init_mask = given_masks[0]
        object_ids = init_mask.unique().tolist()
        if 0 in object_ids:
            object_ids.remove(0)

        # create state for each object
        state = {}
        for k in object_ids:
            init_seg = torch.cat([init_mask != k, init_mask == k], dim=1).float()
            init_seg_16 = F.avg_pool2d(init_seg, 16)
            state[k] = self.vos.get_init_state(norm_key, init_seg_16)

        # spatial distance scores
        sds = self.dsf(dist.view(-1, 1)).view(1, -1, H0 // 16, W0 // 16)

        # initial frame process
        seg_lst = [given_masks[0]]
        frames_to_process = range(1, L)

        # subsequent frames inference
        for i in frames_to_process:

            # extract features
            feats = self.vos.encoder(video_frames[i])
            norm_key = self.vos.matcher.get_norm_key(feats['s16'])

            # inference
            segscore = self.vos(feats, norm_key, state, object_ids, sds)
            predicted_seg = {k: F.softmax(segscore[k], dim=1) for k in object_ids}

            # detect new object
            if given_masks[i] is not None:
                new_object_ids = given_masks[i].unique().tolist()
                if 0 in new_object_ids:
                    new_object_ids.remove(0)
                for new_k in new_object_ids:
                    init_seg = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                    init_seg_16 = F.avg_pool2d(init_seg, 16)
                    state[new_k] = self.vos.get_init_state(norm_key, init_seg_16)
                    predicted_seg[new_k] = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                object_ids = object_ids + new_object_ids

            # aggregate scores
            output_seg, aggregated_seg = softmax_aggregate(predicted_seg, object_ids)
            update_seg_16 = {k: F.avg_pool2d(aggregated_seg[k], 16) for k in object_ids}

            # update state
            if i < L - 1:
                state = self.vos.update(norm_key, update_seg_16, state, object_ids)

            # generate hard masks
            if given_masks[i] is not None:
                output_seg[given_masks[i] != 0] = 0
                seg_lst.append(output_seg + given_masks[i])
            else:
                if val_frame_ids is not None:
                    adjusted_i = i + val_frame_ids[0]
                    if adjusted_i in val_frame_ids:
                        seg_lst.append(output_seg)
                else:
                    seg_lst.append(output_seg)

        # generate output
        output = {}
        output['segs'] = torch.stack(seg_lst, dim=1)
        output['segs'] = unpad(output['segs'], required_padding)
        return output
