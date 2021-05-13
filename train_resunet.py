from model.resnet_unet import ResNet_UNet
from src.utils import *
from data_loader.unetdataset import UnetDataSet

import os
import torch
import torch.optim as optim
import configparser

from torch.autograd import Variable
from torch.utils.data import DataLoader


# multi-GPU
# device_ids = [0, 1, 2, 3]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# *********************** hyper parameter  ***********************

config = configparser.ConfigParser()
config.read('conf.text')
data_label_dir = config.get('data', 'data_label_dir')
save_dir = config.get('data', 'save_dir')

learning_rate = config.getfloat('training', 'learning_rate')
batch_size = config.getint('training', 'batch_size')
epochs = config.getint('training', 'epochs')
begin_epoch = config.getint('training', 'begin_epoch')

cuda = torch.cuda.is_available()
# cuda = False

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# *********************** Build dataset ***********************
train_data = UnetDataSet(data_label_dir=data_label_dir)
print('Train dataset total number of images sequence is ----' + str(len(train_data)))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=False)

# *********************** Build model ***********************

net = ResNet_UNet(in_chans=3, n_classes=3, up_mode='upsample')
# net = VGG_19bn_8s(n_class=3)
if cuda:
    net = net.to(device)
    # net.cuda()
# if cuda:
#     net = net.cuda(device_ids[0])
#     net = nn.DataParallel(net, device_ids=device_ids)


# if begin_epoch > 0:
#     save_path = 'ckpt/model_epoch' + str(begin_epoch) + '.pth'
#     state_dict = torch.load(save_path)
#     net.load_state_dict(state_dict)


def train(visual_train=False):
    # *********************** initialize optimizer ***********************
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
    criterion = cross_entropy2d_loss()  # loss function pixel-wised softmax cross entropy
    if cuda:
        criterion = criterion.to(device)
    net.train()
    for epoch in range(begin_epoch, epochs + 1):
        print('epoch....................' + str(epoch))
        # for step, (image, label_map, center_map, imgs) in enumerate(train_dataset):
        for step, (im_input, im_label, image_input_dir, image_label_dir) in enumerate(train_dataset):
            image = Variable(im_input.to(device) if cuda else im_input)  # 4D Tensor
            # Batch_size  *  3  *  width(256)  *  height(256)
            label = Variable((255 * im_label).to(device) if cuda else (255 * im_label))
            # Batch_size  *  1 *  width(256)  *  height(256)
            label = torch.squeeze(label, dim=1).long()
            # Batch_size  *  width(256)  *  height(256)
            # if visual_train:
            #     import torchvision.transforms as transforms
            #     import matplotlib.pyplot as plt
            #     image_show = image.cpu().float()
            #     image_show = transforms.ToPILImage()(torch.squeeze(image_show, dim=0))
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(image_show)
            #     # plt.pause(5)
            #     label_show = torch.squeeze(label.cpu().float(), dim=0)
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(label_show)
            #     plt.show()
            optimizer.zero_grad()
            pred = net(image)  # 4D tensor:  batch size * c * 256 * 256

            if visual_train and epoch % 10 == 0:
                import torchvision.transforms as transforms
                import matplotlib.pyplot as plt
                image_show = image.cpu().float()
                image_show = transforms.ToPILImage()(torch.squeeze(image_show, dim=0))
                plt.subplot(1, 3, 1)
                plt.imshow(image_show)
                # plt.pause(5)
                label_show = torch.squeeze(label.cpu().float(), dim=0)
                plt.subplot(1, 3, 2)
                plt.imshow(label_show)

                result_show = pred[0]
                c, w, h = result_show.size()
                result = np.zeros((w, h), dtype=float)
                for i in range(w):
                    for j in range(h):
                        cls_result = result_show[:, i, j]
                        result[i][j] = np.argmax(cls_result.cpu().detach().numpy())
                plt.subplot(1, 3, 3)
                plt.imshow(result)
                plt.title(f'res_unet_epoch: {epoch}')
                plt.show()
            # ******************** calculate loss ********************
            loss = criterion(input=pred, target=label)

            # backward
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            print('--step .....' + str(step))
            print('--loss ' + str(float(loss.cpu())))

            # if step % 10 == 0:
            #     print('--step .....' + str(step))
            #     print('--loss ' + str(float(loss.data[0])))

            # if step % 200 == 0:
            #     save_images(label_map[:, 5, :, :, :], pred_6[:, 5, :, :, :], step, epoch, imgs)

        if epoch % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'model_epoch{:d}.pth'.format(epoch)))

    print('train done!')


if __name__ == '__main__':
    train(visual_train=True)
