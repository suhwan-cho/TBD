import utils
import os
import time
import torch
import torch.nn.functional as F


class Evaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.img_saver = utils.ImageSaver()
        self.sdm = utils.DAVISLabels()

    def evaluate_video(self, model, video_name, video_parts, output_path):
        for vos_data in video_parts:
            imgs = vos_data['imgs'].cuda()
            given_masks = [mask.cuda() if mask is not None else None for mask in vos_data['given_masks']]
            files = vos_data['files']
            val_frame_ids = vos_data['val_frame_ids']
            original_imgs = imgs.clone()
            original_given_masks = given_masks.copy()

            # resize to 480p
            resize = False
            H, W = imgs.size(3), imgs.size(4)
            if H > W:
                if W != 480:
                    resize = True
                    ratio = 480 / W
                    imgs = F.interpolate(imgs[0], size=(int(ratio * H), 480), mode='bicubic').unsqueeze(0)
                    for i in range(len(given_masks)):
                        if given_masks[i] is None:
                            continue
                        else:
                            given_masks[i] = F.interpolate(given_masks[i].float(), size=(int(ratio * H), 480), mode='nearest').long()
            else:
                if H != 480:
                    resize = True
                    ratio = 480 / H
                    imgs = F.interpolate(imgs[0], size=(480, int(ratio * W)), mode='bicubic').unsqueeze(0)
                    for i in range(len(given_masks)):
                        if given_masks[i] is None:
                            continue
                        else:
                            given_masks[i] = F.interpolate(given_masks[i].float(), size=(480, int(ratio * W)), mode='nearest').long()

            # back to original size if objects are too small
            if resize:
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
            vos_out = model(imgs, given_masks, dist.unsqueeze(0), val_frame_ids)
            t1 = time.time()

            # save output
            for i in range(len(files)):
                fpath = os.path.join(output_path, video_name, files[i])
                data = ((vos_out['masks'][0, i, 0, :, :].cpu().byte().numpy(), fpath), self.sdm)
                self.img_saver.enqueue(data)
        return t1 - t0, imgs.size(1)

    def evaluate(self, model, output_path):
        model.cuda()
        total_seconds, total_frames = 0, 0
        for video_name, video_parts in self.dataset.get_videos():
            os.makedirs(os.path.join(output_path, video_name), exist_ok=True)
            seconds, frames = self.evaluate_video(model, video_name, video_parts, output_path)
            total_seconds = total_seconds + seconds
            total_frames = total_frames + frames
            print('{} done, {:.1f} fps'.format(video_name, frames / seconds))
        print('total fps: {:.1f}\n'.format(total_frames / total_seconds))
        self.img_saver.kill()
