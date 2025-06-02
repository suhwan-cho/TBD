from evaluation import metrics
from utils import AverageMeter, get_iou
import copy
import numpy
import random
import torch


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_set, save_name, save_step, val_step):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_set = val_set
        self.save_name = save_name
        self.save_step = save_step
        self.val_step = val_step
        self.epoch = 1
        self.best_score = 0
        self.score = 0
        self.stats = {'loss': AverageMeter(), 'iou': AverageMeter()}

        # distance matrix
        h = (384 + 15) // 16
        w = (384 + 15) // 16
        dist = torch.zeros(h * w, h * w).cuda()
        block = torch.zeros(w, w).cuda()
        for i in range(w):
            for j in range(w):
                block[i, j] = (i - j) ** 2
        for i in range(h):
            for j in range(h):
                dist[i * w: (i + 1) * w, j * w: (j + 1) * w] = (block + (i - j) ** 2) ** 0.5
        self.dist = dist

    def train(self, max_epochs):
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.save_step == 0:
                print('saving checkpoint\n')
                self.save_checkpoint()
            if self.score > self.best_score:
                print('new best checkpoint, after epoch {}\n'.format(self.epoch))
                self.save_checkpoint(alt_name='best')
                self.best_score = self.score
        print('finished training!\n', flush=True)

    def train_epoch(self):

        # train
        self.cycle_dataset(mode='train')

        # val
        if self.epoch % self.val_step == 0:
            if self.val_set is not None:
                with torch.no_grad():
                    self.score = self.cycle_dataset(mode='val')

        # update stats
        for stat_value in self.stats.values():
            stat_value.new_epoch()

    def cycle_dataset(self, mode):
        if mode == 'train':
            for vos_data in self.train_loader:
                imgs = vos_data['imgs'].cuda()
                masks = vos_data['masks'].cuda()
                B, L, _, H, W = imgs.size()

                # swap-and-attach augmentation
                if random.random() > 0.8:
                    objects = torch.roll(imgs * masks, dims=0, shifts=1)
                    object_masks = torch.roll(masks, dims=0, shifts=1)
                    imgs = (1 - object_masks) * imgs + object_masks * objects
                    masks = (1 - object_masks) * masks
                gt_masks = masks[:, 1:]

                # if no target -> skip the batch
                skip = False
                for batch in range(B):
                    if len(masks[batch, 0].unique()) != 2:
                        skip = True
                        break
                if skip:
                    continue

                # model run
                vos_out = self.model(imgs, [masks[:, 0]] + (L - 1) * [None], self.dist.unsqueeze(0).repeat(B, 1, 1), None)
                loss = torch.nn.CrossEntropyLoss()(vos_out['scores'].view(B * (L - 1), 2, H, W), gt_masks.reshape(B * (L - 1), H, W))

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss, iou
                self.stats['loss'].update(loss.detach().cpu().item(), B)
                iou = torch.mean(get_iou(vos_out['scores'].view(B * (L - 1), 2, H, W), gt_masks.reshape(B * (L - 1), H, W))[:, 1:])
                self.stats['iou'].update(iou.detach().cpu().item(), B)

            print('[ep{:04d}] loss: {:.5f}, iou: {:.5f}'.format(self.epoch, self.stats['loss'].avg, self.stats['iou'].avg))

        if mode == 'val':
            metrics_res = {}
            metrics_res['J'] = []
            metrics_res['F'] = []
            for video_name, video_parts in self.val_set.get_videos():
                for vos_data in video_parts:
                    imgs = vos_data['imgs'].cuda()
                    given_masks = [mask.cuda() if mask is not None else None for mask in vos_data['given_masks']]
                    masks = vos_data['masks'].cuda()

                    # distance matrix
                    h = (imgs.size(-2) + 15) // 16
                    w = (imgs.size(-1) + 15) // 16
                    dist = torch.zeros(h * w, h * w).cuda()
                    block = torch.zeros(w, w).cuda()
                    for i in range(w):
                        for j in range(w):
                            block[i, j] = (i - j) ** 2
                    for i in range(h):
                        for j in range(h):
                            dist[i * w: (i + 1) * w, j * w: (j + 1) * w] = (block + (i - j) ** 2) ** 0.5

                    # inference
                    vos_out = self.model(imgs, given_masks, dist.unsqueeze(0), None)
                    res_masks = vos_out['masks'][:, 1:-1].squeeze(2)
                    gt_masks = masks[:, 1:-1].squeeze(2)
                    B, L, H, W = res_masks.shape
                    object_ids = numpy.unique(gt_masks.cpu()).tolist()
                    object_ids.remove(0)

                    # evaluate output
                    all_res_masks = numpy.zeros((len(object_ids), L, H, W))
                    all_gt_masks = numpy.zeros((len(object_ids), L, H, W))
                    for k in object_ids:
                        res_masks_k = copy.deepcopy(res_masks).cpu().numpy()
                        res_masks_k[res_masks_k != k] = 0
                        res_masks_k[res_masks_k != 0] = 1
                        all_res_masks[k - 1] = res_masks_k[0]
                        gt_masks_k = copy.deepcopy(gt_masks).cpu().numpy()
                        gt_masks_k[gt_masks_k != k] = 0
                        gt_masks_k[gt_masks_k != 0] = 1
                        all_gt_masks[k - 1] = gt_masks_k[0]

                    # calculate scores
                    j_metrics_res = numpy.zeros(all_gt_masks.shape[:2])
                    f_metrics_res = numpy.zeros(all_gt_masks.shape[:2])
                    for i in range(all_gt_masks.shape[0]):
                        j_metrics_res[i] = metrics.db_eval_iou(all_gt_masks[i], all_res_masks[i])
                        f_metrics_res[i] = metrics.db_eval_boundary(all_gt_masks[i], all_res_masks[i])
                        [JM, _, _] = metrics.db_statistics(j_metrics_res[i])
                        metrics_res['J'].append(JM)
                        [FM, _, _] = metrics.db_statistics(f_metrics_res[i])
                        metrics_res['F'].append(FM)

            # gather scores
            J, F = metrics_res['J'], metrics_res['F']
            final_mean = (numpy.mean(J) + numpy.mean(F)) / 2.
            print('[ep{:04d}] J&F score: {:.5f}\n'.format(self.epoch, final_mean))
            return final_mean

    def save_checkpoint(self, alt_name=None):
        if alt_name is not None:
            file_path = 'weights/{}_{}.pth'.format(self.save_name, alt_name)
        else:
            file_path = 'weights/{}_{:04d}.pth'.format(self.save_name, self.epoch)
        torch.save(self.model.state_dict(), file_path)

    def load_checkpoint(self, save_name, epoch, device):
        checkpoint_path = 'weights/{}_{:04d}.pth'.format(save_name, epoch)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.epoch = epoch + 1
        print('loaded: {}'.format(checkpoint_path))
