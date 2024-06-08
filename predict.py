import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def get_segment_crop(img, mask, cl=[0]):
    img[~np.isin(mask, cl)] = 0
    return img

def predict_img(net,
                full_img,
                device,
                image_size=(256, 256),
                out_threshold=0.5,
                out_mask_filename='mask.png'):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, image_size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        print(f'Ouput shape: {output.shape}')
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        
        if net.n_classes > 1:
            mask = output.argmax(dim=1)

            # Save all crops masks
            mask_filename = out_mask_filename[:out_mask_filename.rfind('.')]
            for cl, mask_class in enumerate(output[0]):
                mask_reshaped = mask.numpy().reshape((mask.shape[1], mask.shape[2]))
                # Crop each class
                full_img_cropped = get_segment_crop(np.array(full_img), mask=mask_reshaped, cl=[cl])
                full_img_cropped = Image.fromarray(full_img_cropped)
                full_img_cropped.save(f'{mask_filename}-{cl}.png')

                '''
                mask_class = torch.sigmoid(mask_class) > out_threshold

                # Mask without argmax (only using threshold)
                mask_class_without_argmax = Image.fromarray(mask_class.numpy().astype(bool))
                mask_class_without_argmax.save(f'{mask_filename}-{cl}-mask_without_argmax.png')

                # Mask using argmax
                only_mask_class = get_segment_crop(mask_class.numpy().astype(bool), mask=mask_reshaped, cl=[cl])
                only_mask_class = Image.fromarray(only_mask_class)
                only_mask_class.save(f'{mask_filename}-{cl}-mask.png')
                '''

            full_img_cropped = get_segment_crop(np.array(full_img), mask=mask_reshaped, cl=range(1, output.shape[1]))
            full_img_cropped = Image.fromarray(full_img_cropped)
            full_img_cropped.save(f'{mask_filename}-all_class.png')

        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='model/checkpoint_epoch11.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default=['output'],
                        help='Filenames or folder of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--image_size', '-s', type=tuple, default=(256, 256),
                        help='Resize images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
        # Add a color for each class (grayscale)
        interval_colors = 255 / (len(mask_values) - 1)
        for idx, v in enumerate(mask_values):
            mask_values[idx] = int(idx * interval_colors)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if os.path.isdir(args.input[0]):
        filenames = os.listdir(args.input[0])
        in_files = []
        for filename in filenames:
            in_files.append(f'{args.input[0]}/{filename}')
        args.input = in_files

    in_files = args.input

    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           image_size=args.image_size,
                           out_threshold=args.mask_threshold,
                           device=device,
                           out_mask_filename=out_files[i])

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
