from .transforms import *
import os
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split, clip_l, clip_n):
        self.root = root
        with open(os.path.join(root, 'ImageSets', '{}/{}.txt'.format(year, split)), 'r') as f:
            self.video_list = f.read().splitlines()
        self.clip_l = clip_l
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        video_name = random.choice(self.video_list)
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)
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


class TestDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split):
        self.root = root
        self.year = year
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
        with open(os.path.join(self.root, 'ImageSets', self.year, self.split + '.txt'), 'r') as f:
            self.video_list = sorted(f.read().splitlines())
            print('--- DAVIS {} {} loaded for testing ---'.format(self.year, self.split))

    def get_snippet(self, video_name, frame_ids):
        img_path = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_path = os.path.join(self.root, 'Annotations', '480p', video_name)
        imgs = torch.stack([self.read_img(os.path.join(img_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
        given_masks = [self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(0))).unsqueeze(0)] + [None] * (len(frame_ids) - 1)
        files = ['{:05d}.png'.format(i) for i in frame_ids]
        if self.split == 'test-dev':
            return {'imgs': imgs, 'given_masks': given_masks, 'files': files, 'val_frame_ids': None}
        masks = torch.stack([self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))) for i in frame_ids]).unsqueeze(0)
        if self.year == '2016':
            masks = (masks != 0).long()
            given_masks[0] = (given_masks[0] != 0).long()
        return {'imgs': imgs, 'given_masks': given_masks, 'masks': masks, 'files': files, 'val_frame_ids': None}

    def get_video(self, video_name):
        frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'JPEGImages', '480p', video_name))])
        yield self.get_snippet(video_name, frame_ids)

    def get_videos(self):
        for video_name in self.video_list:
            yield video_name, self.get_video(video_name)
