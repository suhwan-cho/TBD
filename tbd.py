import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


# pre-, post-processing modules
def aggregate_objects(pred_seg, object_ids):
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in pred_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logit = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7)
             for n, seg in [(-1, bg_seg)] + list(pred_seg.items())}
    logit_sum = torch.cat(list(logit.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logit[n] / logit_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    mask_tmp = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    pred_mask = torch.zeros_like(mask_tmp)
    for idx, obj_idx in enumerate(object_ids):
        pred_mask[mask_tmp == (idx + 1)] = obj_idx
    return pred_mask, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in enumerate(object_ids)}


def get_padding(h, w, div):
    h_pad = (div - h % div) % div
    w_pad = (div - w % div) % div
    padding = [(w_pad + 1) // 2, w_pad // 2, (h_pad + 1) // 2, h_pad // 2]
    return padding


def attach_padding(imgs, given_masks, padding):
    B, L, C, H, W = imgs.size()
    imgs = imgs.view(B * L, C, H, W)
    imgs = F.pad(imgs, padding, mode='reflect')
    _, _, height, width = imgs.size()
    imgs = imgs.view(B, L, C, height, width)
    given_masks = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in given_masks]
    return imgs, given_masks


def detach_padding(output, padding):
    if isinstance(output, list):
        return [detach_padding(x, padding) for x in output]
    else:
        _, _, _, height, width = output.size()
        return output[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]


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

    def forward(self, img):
        x = (img - self.mean) / self.std
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
        return {'s4': s4, 's8': s8, 's16': s16}


# matching module
class Matcher(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = Conv(in_c, out_c, 1, 1, 0)
        self.short = nn.Parameter(torch.Tensor([1, 1]))
        self.long = nn.Parameter(torch.Tensor([-1, -1]))

    def get_key(self, x):
        x = self.conv(x)
        key = x / x.norm(dim=1, keepdim=True)
        return key

    def get_fine_temp(self, key, seg_16):
        B, _, H, W = key.size()
        key = key.view(B, -1, H * W).transpose(1, 2)
        bg_temp = key * seg_16[:, 0].view(B, H * W, 1)
        fg_temp = key * seg_16[:, 1].view(B, H * W, 1)
        fine_temp = [bg_temp, fg_temp]
        return fine_temp

    def get_coarse_temp(self, fine_temp):
        bg_key_sum = torch.sum(fine_temp[0], dim=1, keepdim=True).clamp(min=1e-7)
        fg_key_sum = torch.sum(fine_temp[1], dim=1, keepdim=True).clamp(min=1e-7)
        bg_temp = bg_key_sum / bg_key_sum.norm(dim=2, keepdim=True)
        fg_temp = fg_key_sum / fg_key_sum.norm(dim=2, keepdim=True)
        coarse_temp = [bg_temp, fg_temp]
        return coarse_temp

    def forward(self, key, sds, state):
        B, _, H, W = key.size()
        key = key.view(B, -1, H * W)
        sds = sds.view(B, H * W, H, W)

        # global matching
        score = torch.bmm(state['global'][0], key).view(B, H * W, H, W)
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = torch.bmm(state['global'][1], key).view(B, H * W, H, W)
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        global_score = torch.cat([bg_score, fg_score], dim=1)

        # local matching
        score = torch.bmm(state['local'][0], key).view(B, H * W, H, W) * sds
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = torch.bmm(state['local'][1], key).view(B, H * W, H, W) * sds
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        local_score = torch.cat([bg_score, fg_score], dim=1)
        fine_score = torch.cat([global_score, local_score], dim=1)

        # overall matching
        bg_score = torch.bmm(state['overall'][0], key).view(B, 1, H, W)
        fg_score = torch.bmm(state['overall'][1], key).view(B, 1, H, W)
        overall_score = torch.cat([bg_score, fg_score], dim=1)

        # short-term matching
        bg_score = torch.bmm(state['short'][0], key).view(B, 1, H, W)
        fg_score = torch.bmm(state['short'][1], key).view(B, 1, H, W)
        short_score = torch.cat([bg_score, fg_score], dim=1)

        # long-term matching
        bg_score = torch.bmm(state['long'][0], key).view(B, 1, H, W)
        fg_score = torch.bmm(state['long'][1], key).view(B, 1, H, W)
        long_score = torch.cat([bg_score, fg_score], dim=1)
        coarse_score = torch.cat([overall_score, short_score, long_score], dim=1)

        # collect matching scores
        matching_score = torch.cat([fine_score, coarse_score], dim=1)
        return matching_score


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
        self.deconv3 = DeConv(256, 2, 6, 4, 1)

    def forward(self, feats, matching_score, prev_seg_16):
        x = torch.cat([self.conv1(feats['s16']), matching_score, prev_seg_16], dim=1)
        s8 = self.deconv1(self.cbam1(self.blend1(x)))
        x = torch.cat([self.conv2(feats['s8']), s8], dim=1)
        s4 = self.deconv2(self.cbam2(self.blend2(x)))
        x = torch.cat([self.conv3(feats['s4']), s4], dim=1)
        final_score = self.deconv3(self.cbam3(self.blend3(x)))
        return final_score


# VOS model
class VOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.matcher = Matcher(1024, 512)
        self.decoder = Decoder()

    def get_init_state(self, key, given_seg):

        # get prev seg and seg sum
        state = {}
        given_seg_16 = F.avg_pool2d(given_seg, 16)
        state['prev_seg_16'] = given_seg_16
        state['seg_sum'] = torch.sum(given_seg_16, dim=[2, 3]).unsqueeze(2)

        # get fine matching templates
        fine_temp = self.matcher.get_fine_temp(key, given_seg_16)
        state['global'] = fine_temp
        state['local'] = fine_temp

        # get coarse matching templates
        coarse_temp = self.matcher.get_coarse_temp(fine_temp)
        state['overall'] = coarse_temp
        state['short'] = coarse_temp
        state['long'] = coarse_temp
        return state

    def update_state(self, key, pred_seg, state):

        # update prev seg and seg sum
        pred_seg_16 = F.avg_pool2d(pred_seg, 16)
        state['prev_seg_16'] = pred_seg_16
        seg_sum = torch.sum(pred_seg_16, dim=[2, 3]).unsqueeze(2)
        state['seg_sum'] = state['seg_sum'] + seg_sum

        # update local matching template
        fine_temp = self.matcher.get_fine_temp(key, pred_seg_16)
        state['local'] = fine_temp

        # update overall matching template
        coarse_temp = self.matcher.get_coarse_temp(fine_temp)
        dynamic_inertia = 1 - seg_sum / state['seg_sum']
        bg_temp = dynamic_inertia[:, :1] * state['overall'][0] + (1 - dynamic_inertia[:, :1]) * coarse_temp[0]
        fg_temp = dynamic_inertia[:, 1:] * state['overall'][1] + (1 - dynamic_inertia[:, 1:]) * coarse_temp[1]
        bg_temp = bg_temp / bg_temp.norm(dim=2, keepdim=True)
        fg_temp = fg_temp / fg_temp.norm(dim=2, keepdim=True)
        state['overall'] = [bg_temp, fg_temp]

        # update short-term matching template
        short_inertia = 1 / (1 + torch.exp(self.matcher.short))
        bg_temp = short_inertia[0] * state['short'][0] + (1 - short_inertia[0]) * coarse_temp[0]
        fg_temp = short_inertia[1] * state['short'][1] + (1 - short_inertia[1]) * coarse_temp[1]
        bg_temp = bg_temp / bg_temp.norm(dim=2, keepdim=True)
        fg_temp = fg_temp / fg_temp.norm(dim=2, keepdim=True)
        state['short'] = [bg_temp, fg_temp]

        # update long-term matching template
        long_inertia = 1 / (1 + torch.exp(self.matcher.long))
        bg_temp = long_inertia[0] * state['long'][0] + (1 - long_inertia[0]) * coarse_temp[0]
        fg_temp = long_inertia[1] * state['long'][1] + (1 - long_inertia[1]) * coarse_temp[1]
        bg_temp = bg_temp / bg_temp.norm(dim=2, keepdim=True)
        fg_temp = fg_temp / fg_temp.norm(dim=2, keepdim=True)
        state['long'] = [bg_temp, fg_temp]
        return state

    def forward(self, feats, key, sds, state):

        # get final score
        matching_score = self.matcher(key, sds, state)
        final_score = self.decoder(feats, matching_score, state['prev_seg_16'])
        return final_score


# TBD model
class TBD(nn.Module):
    def __init__(self):
        super().__init__()
        self.vos = VOS()
        self.dsf = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())

    def forward(self, imgs, given_masks, dist, val_frame_ids):

        # basic setting
        B, L, _, H, W = imgs.size()
        padding = get_padding(H, W, 16)
        if tuple(padding) != (0, 0, 0, 0):
            imgs, given_masks = attach_padding(imgs, given_masks, padding)

        # initial frame
        object_ids = given_masks[0].unique().tolist()
        if 0 in object_ids:
            object_ids.remove(0)
        score_lst = []
        mask_lst = [given_masks[0]]

        # initial frame embedding
        with torch.no_grad():
            feats = self.vos.encoder(imgs[:, 0])
        key = self.vos.matcher.get_key(feats['s16'])

        # create state for each object
        state = {}
        for k in object_ids:
            given_seg = torch.cat([given_masks[0] != k, given_masks[0] == k], dim=1).float()
            state[k] = self.vos.get_init_state(key, given_seg)

        # calculate spatial distance scores
        sds = self.dsf(dist.view(-1, 1))

        # subsequent frames
        for i in range(1, L):

            # query frame embedding
            with torch.no_grad():
                feats = self.vos.encoder(imgs[:, i])
            key = self.vos.matcher.get_key(feats['s16'])

            # query frame prediction
            pred_seg = {}
            for k in object_ids:
                final_score = self.vos(feats, key, sds, state[k])
                pred_seg[k] = torch.softmax(final_score, dim=1)

            # detect new object
            if given_masks[i] is not None:
                new_object_ids = given_masks[i].unique().tolist()
                if 0 in new_object_ids:
                    new_object_ids.remove(0)
                for new_k in new_object_ids:
                    given_seg = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                    state[new_k] = self.vos.get_init_state(key, given_seg)
                    pred_seg[new_k] = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                object_ids = object_ids + new_object_ids

            # aggregate objects
            if B == 1:
                pred_mask, pred_seg = aggregate_objects(pred_seg, object_ids)

            # update state
            if i < L - 1:
                for k in object_ids:
                    state[k] = self.vos.update_state(key, pred_seg[k], state[k])

            # store soft scores
            if B != 1:
                score_lst.append(final_score)

            # store hard masks
            if B == 1:
                if given_masks[i] is not None:
                    pred_mask[given_masks[i] != 0] = 0
                    mask_lst.append(pred_mask + given_masks[i])
                else:
                    if val_frame_ids is not None:
                        if val_frame_ids[0] + i in val_frame_ids:
                            mask_lst.append(pred_mask)
                    else:
                        mask_lst.append(pred_mask)

        # generate output
        output = {}
        if B != 1:
            output['scores'] = torch.stack(score_lst, dim=1)
            output['scores'] = detach_padding(output['scores'], padding)
        if B == 1:
            output['masks'] = torch.stack(mask_lst, dim=1)
            output['masks'] = detach_padding(output['masks'], padding)
        return output
