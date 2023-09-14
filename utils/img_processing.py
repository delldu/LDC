import os

import cv2
import numpy as np
import torch
import todos
import pdb

def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def count_parameters(model=None):
    if model is not None:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        print("Error counting model parameters line 32 img_processing.py")
        raise NotImplementedError


def save_image_batch_to_disk(tensor, output_dir, file_names, img_shape=None):
    # len(tensor) -- 5
    # output_dir = 'result/BRIND2CLASSIC'
    # file_names = ['bird.png']
    # img_shape = [tensor([846]), tensor([564])]

    os.makedirs(output_dir, exist_ok=True)
    fuse_name = 'fused'
    av_name = 'avg'

    output_dir_f = os.path.join(output_dir, fuse_name)
    output_dir_a = os.path.join(output_dir, av_name)
    os.makedirs(output_dir_f, exist_ok=True)
    os.makedirs(output_dir_a, exist_ok=True)

    # 255.0 * (1.0 - em_a)
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)

    image_shape = [x.cpu().detach().numpy() for x in img_shape]
    image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

    assert len(image_shape) == len(file_names)

    # pdb.set_trace()

    idx = 0
    for i_shape, file_name in zip(image_shape, file_names):
        tmp = tensor[:, idx, ...]
        tmp = np.squeeze(tmp)

        # Iterate our all 7 NN outputs for a particular image
        preds = []
        fuse_num = tmp.shape[0]-1
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                # ==> pdb.set_trace()
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))

            preds.append(tmp_img)

            if i == fuse_num:
                # print('fuse num',tmp.shape[0], fuse_num, i)
                fuse = tmp_img
                fuse = fuse.astype(np.uint8)

        # Get the mean prediction of all the 7 outputs
        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))


        output_file_name_f = os.path.join(output_dir_f, file_name)
        output_file_name_a = os.path.join(output_dir_a, file_name)

        cv2.imwrite(output_file_name_f, fuse)
        # array [fuse] shape: (846, 564) , min: 7 , max: 255

        # s = average > 200
        # average[s] = 255
        # s = average < 75
        # average[s] = 0

        cv2.imwrite(output_file_name_a, average)
        todos.debug.output_var("average", average)
        # array [average] shape: (846, 564) , min: 9 , max: 255

        idx += 1

