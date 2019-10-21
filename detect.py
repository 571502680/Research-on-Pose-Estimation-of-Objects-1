import os
import torch
import torch.utils.data
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from opt import opt
from models.FastPose import FastPose_SE
from utils.dataset import linemod, occlinemod
from utils.eval import DataLogger, accuracy
from utils.img import flip_v, shuffleLR_v

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    # train_loader_desc = tqdm(train_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(train_loader):
        inps = inps.cuda().requires_grad_()#[32,17,80,64]
        labels = labels.cuda()#[32,17,80,64]
        setMask = setMask.cuda()#[32,17,80,64]
        out = m(inps)#[32,17,80,64]

        loss = criterion(out.mul(setMask), labels)

        acc = accuracy(out.data.mul(setMask),
                       labels.data, train_loader.dataset)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        # train_loader_desc.set_postfix(
        #     loss='%.2e' % lossLogger.avg, acc='%.2f%%' % (accLogger.avg * 100))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar(
            'Train/Loss', lossLogger.avg, opt.trainIters)
        writer.add_scalar(
            'Train/Acc', accLogger.avg, opt.trainIters)

    # train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    # val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(val_loader):
        inps = inps.cuda()
        labels = labels.cuda()
        setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            loss = criterion(out.mul(setMask), labels)

            flip_out = m(flip_v(inps, cuda=True))
            flip_out = flip_v(shuffleLR_v(
                flip_out, val_loader.dataset, cuda=True), cuda=True)

            out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar(
            'Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar(
            'Valid/Acc', accLogger.avg, opt.valIters)

        # val_loader_desc.set_description(
        #     'loss: {loss:.8f} | acc: {acc:.2f}'.format(
        #         loss=lossLogger.avg,
        #         acc=accLogger.avg * 100)
        # )
    #     val_loader_desc.set_postfix(
    #         loss='%.2e' % lossLogger.avg, acc='%.2f%%' % (accLogger.avg * 100))

    # val_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def main():
    # Model Initialize
    expID = '%s_%s_%s_%s%s' % (opt.seq, opt.nClasses, opt.kptype, opt.datatype,
                               '_dpg' if opt.addDPG else '')
    print(expID)

    m = FastPose_SE().cuda()

    if opt.loadModel:
        print('[LOG] Loading model from {}'.format(opt.loadModel))
        m.load_state_dict(torch.load(opt.loadModel))
        if not os.path.exists("/home/common/liqi/data/LINEMOD_6D/LM6d_origin//exp/{}/{}".format(opt.dataset, expID)):
            try:
                os.mkdir("/home/common/liqi/data/LINEMOD_6D/LM6d_origin//exp/{}/{}".format(opt.dataset, expID))
            except FileNotFoundError:
                os.mkdir("/home/common/liqi/data/LINEMOD_6D/LM6d_origin//exp/{}".format(opt.dataset))
                os.mkdir("/home/common/liqi/data/LINEMOD_6D/LM6d_origin//exp/{}/{}".format(opt.dataset, expID))
    else:
        print('[LOG] Create new model')
        if not os.path.exists("/home/common/liqi/data/LINEMOD_6D/LM6d_origin//exp/{}/{}".format(opt.dataset, expID)):
            try:
                os.mkdir("/home/common/liqi/data/LINEMOD_6D/LM6d_origin/exp/{}/{}".format(opt.dataset, expID))
            except FileNotFoundError:
                os.mkdir("/home/common/liqi/data/LINEMOD_6D/LM6d_origin/exp/{}".format(opt.dataset))
                os.mkdir("/home/common/liqi/data/LINEMOD_6D/LM6d_origin/exp/{}/{}".format(opt.dataset, expID))

    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=opt.LR)
    writer = SummaryWriter('tensorboard/{}/{}'.format(opt.dataset, expID))

    # Prepare Dataset
    if opt.dataset == 'linemod':
        train_dataset = linemod.Linemod(train=True)
        val_dataset = linemod.Linemod(train=False)
    elif opt.dataset == 'occlinemod':
        train_dataset = occlinemod.OcclusionLinemod(
            root='/home/penggao/projects/pose/kp6d/keypoint/data/occ',
            train=True
        )
        val_dataset = occlinemod.OcclusionLinemod(
            root='/home/penggao/projects/pose/kp6d/keypoint/data/occ',
            train=False
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.trainBatch, shuffle=True, num_workers=opt.nThreads, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.validBatch, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

    m = torch.nn.DataParallel(m).cuda()
    best_valid_acc = 0
    best_epoch = 0
    for i in trange(opt.nEpochs):
        opt.epoch = i

        # print("\n[LOG] Epoch %d" % i)
        loss, acc = train(train_loader, m, criterion, optimizer, writer)

        opt.acc = acc
        opt.loss = loss

        if i % 10 == 0:
            loss, acc = valid(val_loader, m, criterion, optimizer, writer)
            save_path = '/home/common/liqi/data/LINEMOD_6D/LM6d_origin/exp/{}/{}/model_{}.pkl'.format(
                opt.dataset, expID, opt.epoch
            )
            best_path = '/home/common/liqi/data/LINEMOD_6D/LM6d_origin/exp/{}/{}/model_best.pkl'.format(
                opt.dataset, expID
            )
            # torch.save(m.module.state_dict(), save_path)

            if acc > best_valid_acc:
                best_epoch = i
                best_valid_acc = acc
                torch.save(m.module.state_dict(), best_path)
                # print('[LOG] Epoch %d is the best with accuracy %.2f%%!' %
                    #   (best_epoch, best_valid_acc * 100))

    writer.close()


if __name__ == '__main__':
    main()
