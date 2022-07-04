
import scipy.ndimage
import os
import PIL.Image

from scipy.spatial import ConvexHull
import cv2
import imageio
import numpy as np
from face_alignment import FaceAlignment, LandmarksType


fa = FaceAlignment(LandmarksType._2D, device='cuda:0')

def alpha_blend_images(generated_image, real_image, mask ):
    im_out = ((mask) * real_image + (1-mask) * generated_image)
    return im_out

def create_mask(land, img):
	mask = np.zeros(img.shape[:-1])
	hull = ConvexHull(land[0])
	ps_ = np.int32(land[0][hull.vertices])
	msk = cv2.fillConvexPoly(mask, ps_, (1)).reshape((*img.shape[:-1], 1))
	if len(img.shape) < 3 or img.shape[2] == 1:
		mask = msk
	else:
		mask = np.concatenate((msk, msk, msk), axis=2)
	return mask


def mask_face(land, img, negative=False, return_mask = False):

    mask = create_mask(land, img)
    if negative:
        mask = np.logical_not(mask)
    out = np.multiply(img, mask)
    if return_mask:
        
        return out, mask
    return out





def is_image_file(filename, extensions=['.jpg', '.jpeg', '.png']):
    return any(filename.endswith(e) for e in extensions)

def filter_images(file_list):
    image_list = []
    for fn in file_list:
        if is_image_file(fn):
            image_list.append(fn)
    return image_list


def load_and_split_image(path, return_mask = False):

    filename_array = path.split(os.path.sep)

    hair_path = filename_array.copy()
    hair_path.insert(-1, 'hair')
    if filename_array[0] == '':
        hair_path.insert(0, '/')
    face_path = filename_array.copy()
    face_path.insert(-1, 'inner_face')
    if filename_array[0] == '':
        face_path.insert(0, '/')
    hair_path_with_filename = os.path.join(*hair_path)
    hair_path = os.path.join(*hair_path[:-1])
    face_path_with_filename = os.path.join(*face_path)
    face_path = os.path.join(*face_path[:-1])


    if os.path.exists(hair_path_with_filename):
        inner_f_im = imageio.imread(face_path_with_filename)
        hair_im = imageio.imread(hair_path_with_filename)
        if return_mask:
            mask = (inner_f_im != 0).astype(float)
            return inner_f_im, hair_im, mask
        return inner_f_im, hair_im
    else:
        os.makedirs(hair_path, exist_ok=True)
        os.makedirs(face_path, exist_ok=True)
        img = imageio.imread(path)
        if not return_mask:
            inner_f_im, hair_im = split_image(img, return_mask)
        else:
            inner_f_im, hair_im, mask = split_image(img, return_mask)
        imageio.imwrite(hair_path_with_filename, hair_im)
        imageio.imwrite(face_path_with_filename, inner_f_im)

    if return_mask:
        return inner_f_im, hair_im, mask
    return inner_f_im, hair_im

def gaussian_filter(im, sigma=3.):
    return scipy.ndimage.gaussian_filter(im, sigma)



def split_image(img, return_mask = False):
    img = img[:,:,:3]
    
    img,_ = align_face_npy_with_params(img, 256, True)

    land = fa.get_landmarks(img)
    if land is not None:

        if return_mask:
            inner_f_im, mask = mask_face(land, img, return_mask= True)
        else:
            inner_f_im = mask_face(land, img, return_mask = False)
        hair_im = mask_face(land, img, negative=True)
    else:
        if return_mask:
            return None,None, None
        return None, None

    if return_mask:
        return inner_f_im, hair_im, mask
    return inner_f_im, hair_im

def prepare_dataset(path):
    image_files = filter_images(os.listdir(path))
    os.makedirs(os.path.join(path,'hair'),exist_ok=True)
    os.makedirs(os.path.join(path,'inner_face'),exist_ok=True)

    for fil in image_files:
        inner_f_im, hair_im = load_and_split_image(os.path.join(path,fil))
        if inner_f_im is None:
            continue
        name = fil
        imageio.imwrite(os.path.join(path,'hair',name), np.uint8(hair_im))
        imageio.imwrite(os.path.join(path,'inner_face', name),np.uint8(inner_f_im))

    print(f'data prepared in {path}')

import torch

def numpy_uint8_to_torch(x, device = 'cuda:0', normalize=True):
    
    if type(x) == list:
        x = np.array(x)

    assert x.dtype == np.uint8, f"Wrong type {x.dtype} (expected uint8)"
    
    x = x[..., :3]

    tensor = torch.tensor(x.transpose((2, 0, 1)), device=device, dtype=torch.float32, requires_grad=False).unsqueeze(0)

    tensor.div_(255.)
    # Normalize to [-1, 1].
    if normalize:
        tensor.add_(-0.5).mul_(2.)
    return tensor

def torch_to_numpy_uint8(x, correct_range = False):
    if not correct_range:
        x = ((x.clamp(-1,1)+1)/2)*255
    else:
        x = x * 255
    
    x = x.detach().cpu().numpy()
    x = np.concatenate(x, axis = 2)
    x = x.transpose((1,2,0)).astype(np.uint8)
    return x





# brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
# author: lzhbrian (https://lzhbrian.me)
# date: 2020.1.5
# note: code is heavily borrowed from 
#     https://github.com/NVlabs/ffhq-dataset
#     http://dlib.net/face_landmark_detection.py.html
# requirements:
#     apt install cmake
#     conda install Pillow numpy scipy
#     pip install dlib
#     # download face landmark model from: 
#    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

from PIL import Image
import dlib
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
def get_landmark_npy(img, return_none_with_no_face = False):
    """get landmark with dlib
        :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    if len(dets) == 0:
        if return_none_with_no_face:
            return None
        else:
            raise RuntimeError("No faces found")

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
    

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm

def align_face_npy_with_params(img, output_size=1024, return_none_with_no_face = False):
    lm = get_landmark_npy(img, return_none_with_no_face = return_none_with_no_face)
    if return_none_with_no_face and lm is None:
        return None, None
    
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2


    img = Image.fromarray(img)

    transform_size=4096
    enable_padding=True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    shrunk_image = img

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    actual_crop = (0, 0, 0, 0)
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        actual_crop = crop
        img = img.crop(crop)
        quad -= crop[0:2]

    # # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    actual_padding = (0, 0, 0, 0)
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        actual_padding = pad
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    padded_img = img

    # # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return np.array(img), [shrink, actual_crop, actual_padding, quad, padded_img, shrunk_image]

