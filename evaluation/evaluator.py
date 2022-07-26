import os
import time
import torch
import utils
import torch.nn.functional as F


class Evaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.imsavehlp = utils.ImageSaveHelper()
        self.sdm = utils.ReadSaveDAVISChallengeLabels()

    def read_video_part(self, video_part):
        imgs = video_part['imgs'].cuda()
        given_masks = [mask.cuda() if mask is not None else None for mask in video_part['given_masks']]
        fnames = video_part['fnames']
        val_frame_ids = video_part['val_frame_ids']
        return imgs, given_masks, fnames, val_frame_ids

    def evaluate_video(self, model, seqname, video_parts, output_path):
        for video_part in video_parts:
            imgs, given_masks, fnames, val_frame_ids = self.read_video_part(video_part)
            original_imgs = imgs.clone()
            original_given_masks = given_masks.copy()

            # change to 480p
            resized = False
            H, W = imgs.size(3), imgs.size(4)
            if H > W:
                if W != 480:
                    resized = True
                    ratio = 480 / W
                    imgs = F.interpolate(imgs[0], size=(int(ratio * H), 480), mode='bicubic', align_corners=False).unsqueeze(0)
                    for i in range(len(given_masks)):
                        if given_masks[i] is None:
                            continue
                        else:
                            given_masks[i] = F.interpolate(given_masks[i].float(), size=(int(ratio * H), 480), mode='nearest').long()
            if H <= W:
                if H != 480:
                    resized = True
                    ratio = 480 / H
                    imgs = F.interpolate(imgs[0], size=(480, int(ratio * W)), mode='bicubic', align_corners=False).unsqueeze(0)
                    for i in range(len(given_masks)):
                        if given_masks[i] is None:
                            continue
                        else:
                            given_masks[i] = F.interpolate(given_masks[i].float(), size=(480, int(ratio * W)), mode='nearest').long()

            # back to original size if objects are too small
            if resized:
                tiny_obj = 0
                for i in range(len(given_masks)):
                    if given_masks[i] is None:
                        continue
                    else:
                        object_ids = given_masks[i].unique().tolist()
                        if 0 in object_ids:
                            object_ids.remove(0)
                        for obj_idx in object_ids:
                            if len(given_masks[i][given_masks[i] == obj_idx]) < 1000:
                                tiny_obj += 1
                if tiny_obj > 0:
                    imgs = original_imgs
                    given_masks = original_given_masks

            # distance matrix
            h = (imgs.size(3) + 15) // 16
            w = (imgs.size(4) + 15) // 16
            dist = torch.zeros(h * w, h * w).cuda()
            block = torch.zeros(w, w).cuda()
            for i in range(w):
                for j in range(w):
                    block[i, j] = (i - j) ** 2
            for i in range(h):
                for j in range(h):
                    dist[i * w: (i + 1) * w, j * w: (j + 1) * w] = (block + (i - j) ** 2) ** 0.5

            # inference
            t0 = time.time()
            vos_out = model(imgs, given_masks, dist, val_frame_ids)
            t1 = time.time()

            for idx in range(len(fnames)):
                fpath = os.path.join(output_path, seqname, fnames[idx])
                data = ((vos_out['segs'][0, idx, 0, :, :].cpu().byte().numpy(), fpath), self.sdm)
                self.imsavehlp.enqueue(data)
        return t1-t0, imgs.size(1)

    def evaluate(self, model, output_path):
        model.cuda()
        model.eval()
        with torch.no_grad():
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tot_time, tot_frames = 0.0, 0.0
            for seqname, video_parts in self.dataset.get_video_generator():
                savepath = os.path.join(output_path, seqname)
                os.makedirs(savepath, exist_ok=True)
                time_elapsed, frames = self.evaluate_video(model, seqname, video_parts, output_path)
                tot_time += time_elapsed
                tot_frames += frames
                print(seqname, 'fps:{}, frames:{}, time:{}'.format(frames / time_elapsed, frames, time_elapsed))
            print('\nTotal fps:{}\n\n'.format(tot_frames/tot_time))
        self.imsavehlp.kill()
