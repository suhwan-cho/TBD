from .transforms import *
import os
from PIL import Image
from glob import glob
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainYTVOS(torch.utils.data.Dataset):
    def __init__(self, root, split, clip_l, clip_n):
        self.root = root
        self.split = split
        with open(os.path.join(root, 'ImageSets', '{}.txt'.format(split)), 'r') as f:
            self.video_list = f.read().splitlines()
        self.clip_l = clip_l
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        video_name = random.choice(self.video_list)
        img_dir = os.path.join(self.root, self.split, 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, self.split, 'Annotations', video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        # reverse video
        if random.random() > 0.5:
            img_list.reverse()
            mask_list.reverse()

        # get flip param
        h_flip = False
        if random.random() > 0.5:
            h_flip = True
        v_flip = False
        if random.random() > 0.5:
            v_flip = True

        # select training frames
        all_frames = list(range(len(img_list)))
        selected_frames = random.sample(all_frames, 1)
        for _ in range(self.clip_l - 1):
            if selected_frames[-1] + 1 > all_frames[-1]:
                selected_frames.append(selected_frames[-1])
            else:
                selected_frames.append(selected_frames[-1] + 1)

        # generate training snippets
        img_lst = []
        mask_lst = []
        for i, frame_id in enumerate(selected_frames):
            img = Image.open(img_list[frame_id]).convert('RGB')
            mask = Image.open(mask_list[frame_id]).convert('P')

            # resize to 480p
            W, H = img.size[0], img.size[1]
            if H > W:
                if W != 480:
                    ratio = 480 / W
                    img = img.resize((480, int(ratio * H)), Image.BICUBIC)
                    mask = mask.resize((480, int(ratio * H)), Image.NEAREST)
            else:
                if H != 480:
                    ratio = 480 / H
                    img = img.resize((int(ratio * W), 480), Image.BICUBIC)
                    mask = mask.resize((int(ratio * W), 480), Image.NEAREST)

            # joint flip
            if h_flip:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if v_flip:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # random affine transformation
            ret = random_affine_params(degree=5, scale_ranges=(0.95, 1.05), shear=5)
            if i == 0:
                img = TF.affine(img, *ret, Image.BICUBIC)
                mask = TF.affine(mask, *ret, Image.NEAREST)
                old_ret = ret
            else:
                new_ret = (ret[0] + old_ret[0], (0, 0), ret[2] * old_ret[2], (ret[3][0] + old_ret[3][0], ret[3][1] + old_ret[3][1]))
                img = TF.affine(img, *new_ret, Image.BICUBIC)
                mask = TF.affine(mask, *new_ret, Image.NEAREST)
                old_ret = new_ret

            # joint balanced random crop
            if i == 0:
                for cnt in range(10):
                    y, x, h, w = random_crop_params(mask, scale=(0.8, 1.0))
                    temp_mask = self.to_mask(TF.resized_crop(mask, y, x, h, w, (384, 384), Image.NEAREST))

                    # select one object from reference frame
                    selected_id = 2022
                    possible_obj_ids = temp_mask.unique().tolist()
                    if 0 in possible_obj_ids:
                        possible_obj_ids.remove(0)
                    if len(possible_obj_ids) > 0:
                        selected_id = random.choice(possible_obj_ids)
                    temp_mask[temp_mask != selected_id] = 0
                    temp_mask[temp_mask != 0] = 1

                    # ensure at least 256 FG pixels
                    if len(temp_mask[temp_mask != 0]) >= 256 or cnt == 9:
                        img_lst.append(self.to_tensor(TF.resized_crop(img, y, x, h, w, (384, 384), Image.BICUBIC)))
                        mask_lst.append(temp_mask)
                        break
            else:
                img_lst.append(self.to_tensor(TF.resized_crop(img, y, x, h, w, (384, 384), Image.BICUBIC)))
                temp_mask = self.to_mask(TF.resized_crop(mask, y, x, h, w, (384, 384), Image.NEAREST))
                temp_mask[temp_mask != selected_id] = 0
                temp_mask[temp_mask != 0] = 1
                mask_lst.append(temp_mask)

        # gather all frames
        imgs = torch.stack(img_lst, 0)
        masks = torch.stack(mask_lst, 0)
        return {'imgs': imgs, 'masks': masks}


class TestYTVOS(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.init_data()

    def read_img(self, path):
        pic = Image.open(path).convert('RGB')
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def read_mask(self, path):
        pic = Image.open(path).convert('P')
        transform = LabelToLongTensor()
        return transform(pic)

    def init_data(self):
        if self.split == 'cho_val':
            with open(os.path.join(self.root, 'ImageSets', 'cho_val.txt'), 'r') as f:
                self.video_list = f.read().splitlines()
        if self.split == 'val':
            self.video_list = sorted(os.listdir(os.path.join(self.root, 'valid', 'Annotations')))
        print('--- YTVOS 2018 {} loaded for testing ---'.format(self.split))

    def get_snippet(self, video_name, frame_ids, val_frame_ids):
        if self.split == 'cho_val':
            img_path = os.path.join(self.root, 'train', 'JPEGImages', video_name)
            mask_path = os.path.join(self.root, 'train', 'Annotations', video_name)
            imgs = torch.stack([self.read_img(os.path.join(img_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
            given_masks = [self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(frame_ids[0]))).unsqueeze(0)] + [None] * (len(frame_ids) - 1)
            masks = torch.stack([self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))) for i in frame_ids]).unsqueeze(0)
            files = ['{:05d}.png'.format(i) for i in frame_ids]
            return {'imgs': imgs, 'given_masks': given_masks, 'masks': masks, 'files': files, 'val_frame_ids': val_frame_ids}
        if self.split == 'val':
            img_path = os.path.join(self.root, 'valid_all_frames', 'JPEGImages', video_name)
            mask_path = os.path.join(self.root, 'valid', 'Annotations', video_name)
            imgs = torch.stack([self.read_img(os.path.join(img_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
            given_masks = [self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))).unsqueeze(0)
                           if i in sorted([int(file[:5]) for file in os.listdir(mask_path)]) else None for i in frame_ids]
            files = ['{:05d}.png'.format(i) for i in val_frame_ids]
            return {'imgs': imgs, 'given_masks': given_masks, 'files': files, 'val_frame_ids': val_frame_ids}

    def get_video(self, video_name):
        if self.split == 'cho_val':
            frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'train', 'JPEGImages', video_name))])
            min_frame_id = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'train', 'Annotations', video_name))])[0]
            frame_ids = [i for i in frame_ids if i >= min_frame_id]
            val_frame_ids = None
        if self.split == 'val':
            frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'valid_all_frames', 'JPEGImages', video_name))])
            val_frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'valid', 'JPEGImages', video_name))])
            min_frame_id = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'valid', 'Annotations', video_name))])[0]
            frame_ids = [i for i in frame_ids if i >= min_frame_id]
            val_frame_ids = [i for i in val_frame_ids if i >= min_frame_id]
        yield self.get_snippet(video_name, frame_ids, val_frame_ids)

    def get_videos(self):
        for video_name in self.video_list:
            yield video_name, self.get_video(video_name)
