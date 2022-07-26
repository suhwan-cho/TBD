import os
from torch.utils import data
from .transforms import *


class YTVOS_Test(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.init_data()

    def img_read(self, path):
        pic = Image.open(path)
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def mask_read(self, path):
        pic = Image.open(path)
        transform = LabelToLongTensor()
        label = transform(pic)
        return label

    def init_data(self):
        print('-- YTVOS dataset initialization started.')
        self.all_seqs = os.listdir(os.path.join(self.root, 'valid', 'Annotations'))
        print('{} sequences found in the Annotations directory.\n'.format(len(self.all_seqs)))

    def get_frame_ids(self, seqname):
        val_frame_ids = sorted([int(os.path.splitext(fname)[0]) for fname in os.listdir(os.path.join(self.root, 'valid', 'JPEGImages', seqname))])
        all_frame_ids = sorted([int(os.path.splitext(fname)[0]) for fname in os.listdir(os.path.join(self.root, 'valid_all_frames', 'JPEGImages', seqname))])
        min_mask_idx = min(sorted([int(os.path.splitext(fname)[0]) for fname in os.listdir(os.path.join(self.root, 'valid', 'Annotations', seqname))]))
        val_frame_ids = [idx for idx in val_frame_ids if idx >= min_mask_idx]
        all_frame_ids = [idx for idx in all_frame_ids if idx >= min_mask_idx]
        return val_frame_ids, all_frame_ids

    def get_snippet(self, seqname, frame_ids, val_frame_ids):
        imgs = torch.stack([self.img_read(os.path.join(self.root, 'valid_all_frames', 'JPEGImages', seqname, '{:05d}.jpg'.format(idx)))
                            for idx in frame_ids]).unsqueeze(0)
        given_masks = [self.mask_read(os.path.join(self.root, 'valid', 'Annotations', seqname, '{:05d}.png'.format(idx))).unsqueeze(0)
                       if idx in sorted([int(os.path.splitext(fname)[0]) for fname in os.listdir(os.path.join(self.root, 'valid', 'Annotations', seqname))])
                       else None for idx in frame_ids]
        fnames = ['{:05d}.png'.format(idx) for idx in val_frame_ids]
        return {'imgs': imgs, 'given_masks': given_masks, 'fnames': fnames, 'val_frame_ids': val_frame_ids}

    def get_video(self, seqname):
        val_frame_ids, all_frame_ids = self.get_frame_ids(seqname)
        partitioned_frame_ids = [all_frame_ids[start_idx: start_idx + 2 ** 10] for start_idx in range(0, len(all_frame_ids), 2 ** 10)]
        for frame_ids in partitioned_frame_ids:
            yield self.get_snippet(seqname, frame_ids, val_frame_ids)

    def get_video_generator(self):
        for seqname in self.all_seqs:
            yield (seqname, self.get_video(seqname))
