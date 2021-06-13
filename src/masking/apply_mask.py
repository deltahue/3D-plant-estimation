import os
from argparse import ArgumentParser
import struct
import numpy as np

import cv2
from tqdm import tqdm



def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)

def load_mask(address):
    img=cv2.imread(address, 0)
    return img

def apply_mask(img,mask):
    mask=cv2.resize(mask, (img.shape[1], img.shape[0]))
    img[mask==0]=0
    return img

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', default=\
    '../../data/test/')  
    args = parser.parse_args()
    path = args.data_path
    dense='dense/stereo/'
    depth='depth_maps/'
    normals='normal_maps/'
    prefix = 'mask_'
    sufix = '.png'    
    depth_path = path+dense+depth
    normal_path = path+dense+normals
    depth_orig_path = path+dense+'orig_'+depth
    normal_orig_path = path+dense+'orig_'+normals


    if not os.path.exists(depth_orig_path):
        os.rename(normal_path, normal_orig_path)
        os.rename(depth_path, depth_orig_path)
        os.mkdir(normal_path)
        os.mkdir(depth_path)
    depth_names=[name for name in os.listdir(depth_orig_path) if os.path.isfile(os.path.join(depth_orig_path, name))]
    normal_names=[name for name in os.listdir(normal_orig_path) if os.path.isfile(os.path.join(normal_orig_path, name))]
    names=[name for name in os.listdir(path+'images/') if os.path.isfile(os.path.join(path+'images/', name))]
    
    if not os.path.exists(path +'masks/'):
        print("masks do not exist") 
    print("Masking depth maps")
    for i,filename in enumerate(tqdm(os.listdir(depth_orig_path))):
        depth_map = read_array(depth_orig_path+filename)
        depth_map=cv2.rotate(depth_map, cv2.cv2.ROTATE_90_CLOCKWISE)
        real_name = filename.split(".", 1)
        mask=load_mask(path +'masks/'+ real_name[0]+'.jpg')
        new_depth=apply_mask(depth_map,mask)
        new_depth=cv2.rotate(new_depth, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        write_array(new_depth, depth_path+filename)
    print("Masking normal maps")
    for i,filename in enumerate(tqdm(os.listdir(normal_orig_path))):
        normal_map = read_array(normal_orig_path+filename)
        normal_map=cv2.rotate(normal_map, cv2.cv2.ROTATE_90_CLOCKWISE)
        mask=load_mask(path +'masks/'+ real_name[0]+'.jpg' )
        new_normal=apply_mask(normal_map,mask)
        new_normal=cv2.rotate(new_normal, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        write_array(new_normal, normal_path+filename)

  