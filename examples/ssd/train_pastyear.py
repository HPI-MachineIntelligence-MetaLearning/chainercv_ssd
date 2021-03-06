import argparse
import copy
import numpy as np
import json

import chainer
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

from chainercv.utils import read_image

from os.path import join


LABEL_DICT = {'none': '0',
              'funkturm': '1',
              'berlinerdom': '2',
              'brandenburgertor': '3',
              'fernsehturm': '4',
              'reichstag': '5',
              'rotesrathaus': '6',
              'siegessäule': '7',
              'unknown': '8'
              }


class OldDataset(chainer.dataset.DatasetMixin):

    def __init__(self, json_path, img_folder):
        with open(json_path) as json_file:
            img_index = json.load(json_file)
        self._imgs = []
        for img_id, value in img_index.items():
            if value['annotationStatus'] not in ['ignore', 'none']:
                self._imgs.append(join(img_folder, value['id'] + '.jpg'))

    def __len__(self):
        return len(self._imgs)

    def get_example(self, i):
        with open(self._imgs[i].replace('.jpg', '.json')) as bbox_file:
            bbox_obj = json.load((bbox_file))
            bbox_obj_coord = bbox_obj['boundingBox']
            bbox = [[bbox_obj_coord['x'], bbox_obj_coord['y'],
                    bbox_obj_coord['x'] + bbox_obj_coord['width'],
                    bbox_obj_coord['y'] + bbox_obj_coord['height']]]
        try:
            label = np.stack([LABEL_DICT[bbox_obj['label']]
                              for _ in bbox]).astype(np.int32)
        except KeyError:
            label = np.stack([LABEL_DICT['unknown']
                              for _ in bbox]).astype(np.int32)
        bbox = np.stack(bbox).astype(np.float32)
        img = read_image(self._imgs[i], color=True)
        return img, bbox, label


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    parser.add_argument('--json', required=True)
    parser.add_argument('--imgs', required=True)
    args = parser.parse_args()

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(LABEL_DICT),
            pretrained_model='imagenet')
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(LABEL_DICT),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train = TransformDataset(
        OldDataset(args.json, args.imgs),
        Transform(model.coder, model.insize, model.mean))
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)

    test = OldDataset(args.json, args.imgs)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=(10000, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(120000, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
