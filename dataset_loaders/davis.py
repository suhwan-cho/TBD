import os
from torch.utils import data
from .transforms import *


class DAVIS_Test(torch.utils.data.Dataset):
    def __init__(self, root, year, split):
        self.root = root
        self.year = year
        self.split = split
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
        print('-- DAVIS dataset initialization started.')
        with open(os.path.join(self.root, 'ImageSets', self.year, self.split + '.txt'), 'r') as f:
            self.all_seqs = f.read().splitlines()
            print('{} sequences found in image set \'{}\'\n'.format(len(self.all_seqs), self.split))

    def get_snippet(self, seqname, frame_ids):
        imgs = torch.stack([self.img_read(os.path.join(self.root, 'JPEGImages', '480p', seqname, '{:05d}.jpg'.format(idx)))
                            for idx in frame_ids]).unsqueeze(0)
        given_masks = [self.mask_read(os.path.join(self.root, 'Annotations', '480p', seqname, '{:05d}.png'.format(0))).unsqueeze(0)]
        given_masks += [None] * (len(frame_ids)-1)
        fnames = ['{:05d}.png'.format(idx) for idx in frame_ids]

        if self.split == 'test-dev':
            return {'imgs': imgs, 'given_masks': given_masks, 'fnames': fnames, 'val_frame_ids': None}

        masks = torch.stack([self.mask_read(os.path.join(self.root, 'Annotations', '480p', seqname, '{:05d}.png'.format(idx)))
                             for idx in frame_ids]).squeeze().unsqueeze(0)
        if self.year == '2016':
            masks = (masks != 0).long()
            for i in range(len(given_masks)):
                if given_masks[i] is not None:
                    given_masks[i] = (given_masks[i] != 0).long()
        return {'imgs': imgs, 'given_masks': given_masks, 'masks': masks, 'fnames': fnames, 'val_frame_ids': None}

    def get_video(self, seqname):
        seq_frame_ids = sorted([int(os.path.splitext(fname)[0]) for fname in os.listdir(os.path.join(self.root, 'JPEGImages', '480p', seqname))])
        partitioned_frame_ids = [seq_frame_ids[start_idx: start_idx + 2 ** 10] for start_idx in range(0, len(seq_frame_ids), 2 ** 10)]
        for frame_ids in partitioned_frame_ids:
            yield self.get_snippet(seqname, frame_ids)

    def get_video_generator(self):
        for seqname in self.all_seqs:
            yield (seqname, self.get_video(seqname))
