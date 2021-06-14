import os 
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

import cv2


def normal(data):
    return (data - np.min(data)) / \
      (np.max(data) - np.min(data))

# Sparate RGB image in the three channels 
def sepChannels(img):
    img = np.asarray(img)
    img = img.astype('float32')
    img /= 255.0
    blue = img[:,:,0]
    green = img[:,:,1]
    red= img[:,:,2]
    return blue, green, red 

def exg(img): 
    b, g, r = sepChannels(img) 
    channelSum = b + g + r 
    channelSum_=channelSum
    channelSum[channelSum==0]=1
    b = b/channelSum 
    g = g/channelSum 
    r = r/channelSum 
    exg = 2*g-b-r
    exg[channelSum_==0]=0
    exg = normal(exg)
    return exg 

def mask_exg(address, threshold):
  img = cv2.imread(address)
  size = [384, 512]
  img_res = cv2.resize(img, (size[1], size[0]))
  img_e=exg(img_res)
  img_out=cv2.inRange(img_e, (threshold), (1))
  img_out[img_out>0]=1
  pred=img_out*255 
  return pred,img_out


if __name__=='__main__':
  parser = ArgumentParser()
  parser.add_argument('--data_path', default=\
    '../../data/test/')  
  parser.add_argument('--threshold', default=\
    0.35)  
  args = parser.parse_args()
  path=args.data_path
  threshold=args.threshold
  # Dataset
  img_folder='images/'
  folder = 'masks/'
  folder_vis='vis/'
  img_folder_path=path+img_folder
  folder_path=path+folder

  prefix = 'mask_'
  sufix = '.png'
  # Set to the number of datapoints
  IMG_NR = (len([name for name in os.listdir(img_folder_path) \
    if os.path.isfile(os.path.join(img_folder_path, name))]))  
  names = [name for name in os.listdir(img_folder_path) \
    if os.path.isfile(os.path.join(img_folder_path, name))]

  if not os.path.exists(folder_path):
    os.mkdir(folder_path)  
    
  if not os.path.exists(path+folder_vis):
    os.mkdir(path+folder_vis)
  print("Creating masks")
  for i,filename in enumerate(tqdm(names)):
    if filename.endswith(".png") or filename.endswith(".jpeg")\
       or filename.endswith(".jpg"):
      imagrep,imgrep_vis=mask_exg(img_folder_path+filename,threshold)
      address = folder_path+ filename
      cv2.imwrite(address,imagrep[:,:]*255)
      address = path+folder_vis+filename
      cv2.imwrite(address,imgrep_vis[:,:]*255)
    else:
      print("not valid image files")
    




