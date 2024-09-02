from functools import reduce

import cv2
import torch
from kornia import image_to_tensor, create_meshgrid, tensor_to_image
from matplotlib import pyplot as plt
from torch.nn import functional

from random_adjust import RandomAdjust
import os


def transformed_images(image_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    # initialize transform
    # transforms: 'ep' - elastic + perspective ｜ 'e' - elastic only | 'p' - perspective only
    ra = RandomAdjust({'transforms': 'ep', 'kernel_size': (103, 103), 'sigma': (32, 32), 'distortion_scale': 0.3})
    file_names = os.listdir(image_folder)
    file_names.sort()
    # 处理文件夹中的每张图片
    for filename in file_names:
        if filename.endswith('.png'):  # 可以根据需要调整文件类型
            # 读取并准备图像
            img_path = os.path.join(image_folder, filename)
            #x = image_to_tensor(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).float() / 255
            x = image_to_tensor(cv2.imread(img_path, cv2.IMREAD_COLOR)).float() / 255
            x.unsqueeze_(0)  # [1, 1, h, w]

            # 应用变形
            x_w, _ = ra(x)

            # 保存变形后的图像
            transformed_img_path = os.path.join(output_folder, f'{filename}')
            plt.imshow(tensor_to_image(x_w.squeeze() * 255), cmap='gray')
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(transformed_img_path)
            plt.close()



if __name__ == '__main__':
    
    image_folder = '/home/ykx/reconet/data/MSRS_test/vi'
    output_folder = '/home/ykx/reconet/data/MSRS_test/vi_transformed'
    transformed_images(image_folder, output_folder)


'''

def test_complex():
    # read sample images
    x = image_to_tensor(cv2.imread('/home/ykx/reconet/modules/grid.png', cv2.IMREAD_GRAYSCALE)).float() / 255
    x.unsqueeze_(0)  # [1, 1, h, w]

    # initialize transform
    # transforms: 'ep' - elastic + perspective ｜ 'e' - elastic only | 'p' - perspective only
    ra = RandomAdjust({'transforms': 'ep', 'kernel_size': (103, 103), 'sigma': (32, 32), 'distortion_scale': 0.3})

    # x -> warped x
    x_w, params = ra(x)

    # inverse grid -> raw grid - dp - de
    h, w = x.size()[-2:]
    disp = reduce(lambda i, j: i + j, [v for _, v in params.items()])
    grid = create_meshgrid(h, w, device=x.device).to(x.dtype)

    # warped x -> reduction x
    x_r = functional.grid_sample(x_w, (grid - disp), align_corners=True)

    # merge and display
    # 保存变换后的图像 x_w
    plt.imshow(tensor_to_image(x_w.squeeze() * 255), cmap='gray')
    plt.tight_layout()
    plt.savefig('/home/ykx/reconet/modules/transformed_image.png')
    plt.close()  # 清除当前绘图并关闭窗口

    # 保存逆变换后的图像 x_r
    plt.imshow(tensor_to_image(x_r.squeeze() * 255), cmap='gray')
    plt.tight_layout()
    plt.savefig('/home/ykx/reconet/modules/reversed_image.png')
    plt.close()  # 清除当前绘图并关闭窗口



if __name__ == '__main__':
    test_complex()
'''




