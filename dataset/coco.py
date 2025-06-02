from .transforms import *
import os
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainCOCO(torch.utils.data.Dataset):
    def __init__(self, root, clip_l, clip_n):
        self.root = root
        img_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'Annotations')
        self.img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        self.mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.clip_l = clip_l
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):

        # get flip param
        h_flip = False
        if random.random() > 0.5:
            h_flip = True
        v_flip = False
        if random.random() > 0.5:
            v_flip = True

        # generate training snippets
        img_lst = []
        mask_lst = []
        all_frames = list(range(len(self.img_list)))
        k = random.choice(all_frames)
        for i in range(self.clip_l):
            img = Image.open(self.img_list[k]).convert('RGB')
            mask = Image.open(self.mask_list[k]).convert('P')

            # joint flip
            if h_flip:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if v_flip:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # random affine transformation
            ret = random_affine_params(degree=20, scale_ranges=(0.8, 1.25), shear=20)
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
                    y, x, h, w = random_crop_params(mask, scale=(0.5, 1.0))
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
