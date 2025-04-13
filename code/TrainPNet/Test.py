import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network import Network
from utils.data_val import get_test_loader
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=384, help='testing size')  #
    parser.add_argument('--ration', type=int, default=20, help='ration')
    parser.add_argument('--pth_path', type=str, default='../weight/PNet/')  # this
    parser.add_argument('--test_dataset_path', type=str, default='/Users/ankun/works/edu/dlcv/Noisy-COD/code/weight/PNet/1%')
    parser.add_argument('--save_path', type=str, default="../results/")
    args = parser.parse_args()
    args.pth_path = args.pth_path + str(args.ration) + "%/Net_epoch_best.pth"
    all_dataset_mae = []
    datasets = ['COD10K', "CAMO", "CHAMELEON", "NC4K"]
    with torch.no_grad():
        for _data_name in datasets:
            mae = []
            data_path = args.test_dataset_path + '/{}/'.format(_data_name)
            save_path = args.save_path + "/" + str(args.ration) + "%/" + _data_name  # this
            os.makedirs(save_path, exist_ok=True)

            model = Network()
            weights = torch.load(args.pth_path, map_location=torch.device('cpu'))

            weights_dict = {}
            for k, v in weights.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v

            model.load_state_dict(weights_dict)
            model.eval()

            image_root = '{}'.format(data_path)
            gt_root = '{}/mask/'.format(data_path)
            test_loader = get_test_loader(image_root, gt_root, 128, args.testsize, False, 4)

            for i, (image, gt, [H, W], name) in tqdm.tqdm(enumerate(test_loader, start=1)):
                gt = gt.cpu()
                image = image.cpu()
                result = model(image)
                res = result[4]
                res = res.sigmoid()
                for num in range(len(image)):
                    pre = res[num].squeeze().detach().cpu().numpy()
                    gt_single = gt[num].squeeze().detach().cpu().numpy()
                    pre = cv2.resize(pre, dsize=(H[num].item(), W[num].item()))
                    gt_single = cv2.resize(gt_single, dsize=(H[num].item(), W[num].item()))
                    mae.append(np.mean(np.abs(gt_single - pre)))
                    cv2.imwrite(save_path + '/' + name[num].replace(".jpg", ".png"), pre * 255.)
            print(_data_name, ':', np.mean(mae))
