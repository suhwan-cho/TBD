from dataset import *
import evaluation
from tbd import TBD
from trainer import Trainer
from optparse import OptionParser
import warnings
warnings.filterwarnings('ignore')


parser = OptionParser()
parser.add_option('--train', action='store_true', default=None)
parser.add_option('--test', action='store_true', default=None)
options = parser.parse_args()[0]


def train_coco(model):
    train_set = TrainCOCO('../DB/VOS/COCO', clip_l=5, clip_n=128)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, num_workers=4, pin_memory=True)
    val_set = TestDAVIS('../DB/VOS/DAVIS', '2017', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='coco', save_step=4000, val_step=400)
    trainer.train(8000)


def train_davis(model):
    train_set = TrainDAVIS('../DB/VOS/DAVIS', '2017', 'train', clip_l=10, clip_n=128)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, num_workers=4, pin_memory=True)
    val_set = TestDAVIS('../DB/VOS/DAVIS', '2017', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='coco_davis', save_step=1000, val_step=100)
    trainer.train(4000)


def train_ytvos(model):
    train_set = TrainYTVOS('../DB/VOS/YTVOS18', 'cho_train', clip_l=10, clip_n=128)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, num_workers=4, pin_memory=True)
    val_set = TestYTVOS('../DB/VOS/YTVOS18', 'cho_val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='coco_ytvos', save_step=1000, val_step=100)
    trainer.train(4000)


def test(model):
    datasets = {
        'DAVIS16_val': TestDAVIS('../DB/VOS/DAVIS', '2016', 'val'),
        'DAVIS17_val': TestDAVIS('../DB/VOS/DAVIS', '2017', 'val'),
        'DAVIS17_test-dev': TestDAVIS('../DB/VOS/DAVIS', '2017', 'test-dev'),
        # 'YTVOS18_val': TestYTVOS('../DB/VOS/YTVOS18', 'val')
    }

    for key, dataset in datasets.items():
        evaluator = evaluation.Evaluator(dataset)
        evaluator.evaluate(model, os.path.join('outputs', key))


if __name__ == '__main__':

    # set device
    torch.cuda.set_device(0)

    # define model
    model = TBD().eval()

    # training stage
    if options.train:
        train_coco(model)
        train_davis(model)
        # train_ytvos(model)

    # testing stage
    if options.test:
        model.load_state_dict(torch.load('weights/TBD_davis.pth', map_location='cpu'))
        with torch.no_grad():
            test(model)
