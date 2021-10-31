import numpy as np 
import cv2
import random
def policy_v0():
  """Autoaugment policy that was used in AutoAugment Paper."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
      [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
      [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
      [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
      [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
      [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
      [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
      [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
      [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
      [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
      [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
      [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
      [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
      [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
      [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
      [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
      [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
      [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
      [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
      [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
      [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
      [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
      [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
      [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
      [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
  ]
  return policy




# def parse_level():



def mixup(img1,img2,factor):
    img = img1.astype('float')* factor + img2.astype('float') * (1-factor)
    img = np.clip(img, 0,255)
    img = img.astype('uint8')
    return img

def augment_fliplr(img,level):
    if random.random() < level:
        return np.fliplr(img)
    return img 

def augment_hsv(im, level = None,hgain=0.2, sgain=0.2, vgain=0.2):
    im = im.copy()
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
    return im_hsv
def hist_equalize(im, clahe=True, bgr=True):
    im = im.copy()
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

def solarize(image, threshold=128):
    image = image.copy()
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return np.where(image <= threshold, image, 255 - image)

def posterize(img, bits=3):
    shift = 8 - bits
    # img = img >> shift
    img = np.left_shift(img,shift)
    img = np.right_shift(img,shift)
    return img.astype('uint8')

def adjust_brightness(img,factor=0.5):
    degenerate = np.zeros(img.shape,dtype='uint8')
    img = mixup(img,degenerate,factor)
    return img

def invert(img,level=None):
    return 255-img

def contrast(img,factor=0.5): 
    degenerate = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return mixup(img,degenerate,factor)

def shear_x(img,level=0.4,mode='reflect'):
    M = np.array([[1, level, 0],
                  [0,  1   , 0],
                  [0,  0   , 1]],dtype='float')
    height,width,_ = img.shape
    option_mode ={
        'reflect'  : cv2.BORDER_REPLICATE,
        'constant' : cv2.BORDER_CONSTANT
    }
    mode = option_mode[mode]
    sheared_img = cv2.warpPerspective(img, M, (width, height), borderMode=mode)
    return sheared_img

def shear_y(img,level=0.4,mode='reflect'):
    M = np.array([[1,      0   , 0],
                  [level,  1   , 0],
                  [0,      0   , 1]],dtype='float')
    height,width,_ = img.shape
    option_mode ={
        'reflect'  : cv2.BORDER_REPLICATE,
        'constant' : cv2.BORDER_CONSTANT
    }
    mode = option_mode[mode]
    sheared_img = cv2.warpPerspective(img, M, (width, height), borderMode=mode)
    return sheared_img

def translate_x(img,level,mode='reflect'): 
    height,width,_ = img.shape
    option_mode ={
        'reflect'  : cv2.BORDER_REPLICATE,
        'constant' : cv2.BORDER_CONSTANT
    }
    mode = option_mode[mode]
    translate_pixel = int(width * level)
    M = np.array([[1,      0   , translate_pixel],
                [level,  1   , 0],
                [0,      0   , 1]],dtype='float')
    translate_img = cv2.warpPerspective(img, M, (width, height), borderMode=mode)
    return translate_img

def translate_y(img,level,mode='reflect'): 
    height,width,_ = img.shape
    option_mode ={
        'reflect'  : cv2.BORDER_REPLICATE,
        'constant' : cv2.BORDER_CONSTANT
    }
    mode = option_mode[mode]
    translate_pixel = int(width * level)
    M = np.array([[1,      0   , 0],
                [level,  1   , translate_pixel],
                [0,      0   , 1]],dtype='float')
    translate_img = cv2.warpPerspective(img, M, (width, height), borderMode=mode)
    return translate_img

# def sharpness(img,): 
#     kernel = np.array(
#     [[1, 1, 1], 
#     [1, 5, 1], 
#     [1, 1, 1]], dtype=tf.float32,
#     shape=[3, 3, 1, 1]) / 13.
#     cv2.

def cutout(img,level,**kwargs): 
    img = img.copy()
    height,width ,_ = img.shape 
    padding_size = int(height*level),int(width*level)
    value = kwargs.get('value') 
    cordinate_h = np.random.randint(0,height-padding_size[0])
    cordinate_w = np.random.randint(0,width-padding_size[1])
    img[cordinate_h:cordinate_h+padding_size[0],cordinate_w:cordinate_w+padding_size[1],:] = 255
    return img 

def rotate(image, angle=45, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h),borderMode=cv2.BORDER_REPLICATE)

    return rotated

AUGMENT_FUNCTION = {
    # 'mixup' : mixup,
    'fliplr' : augment_fliplr,
    'augment_hsv': augment_hsv,
    'hist_equalize' : hist_equalize,
    # 'solarize' : solarize,
    # 'posterize': posterize,
    'adjust_brightness': adjust_brightness,
    # 'invert' : invert,
    'contrast': contrast,
    'shearX' : shear_x,
    'shearY' : shear_y,
    'translateX' : translate_x,
    'translateY' : translate_y,
    # 'sharpness' : sharpness,
    'cutout' : cutout,
    'rotate' : rotate
    # 'random_crop':random_crop
}
ARGS_LIMIT = {
    'fliplr' : [0,1],
    'augment_hsv': [0,1],
    'hist_equalize' : [0,1],
    # 'solarize' : solarize,
    # 'posterize': posterize,
    'adjust_brightness': [0.5,1.5],
    # 'invert' : invert,
    'contrast': [0.5,1.5],
    'shearX' : [-0.5,0.5],
    'shearY' : [-0.5,-0.5],
    'translateX' : [-0.4,-0.4],
    'translateY' : [-0.4,-0.4],
    # 'sharpness' : sharpness,
    'cutout' : [0.1,0.3],
    'rotate' : [-45,45]
    # 'random_crop':random_crop
}


def preprocess(img,img_size,padding=True):
    if padding:
        height,width,_ = img.shape 
        delta = height - width 
        if delta > 0:
            img = np.pad(img,[[0,0],[delta//2,delta//2],[0,0]], mode='constant',constant_values =255)
        else:
            img = np.pad(img,[[delta//2,delta//2],[0,0],[0,0]], mode='constant',constant_values =255)
    if isinstance(img_size,int):
        img_size = (img_size,img_size)
    return cv2.resize(img,img_size)

class RandAugment:

    def __init__(self, num_layers=2,):
        self.num_layers = num_layers
        self.policy = list(AUGMENT_FUNCTION.keys()
    
    def __call__(self,img):
        # print(self.policy)
        augmenters = random.choices(self.policy, k=self.num_layers)
        print(augmenters)
        for augmenter in augmenters:
            
            level = random.random()
            print(level)
            # level = 0.5
            min_arg,max_arg = ARGS_LIMIT[augmenter]
            level = min_arg + (max_arg - min_arg) * level
            img = AUGMENT_FUNCTION[augmenter](img,level)
        return img 



if __name__ =='__main__':
    img_org = cv2.imread('/u01/Intern/chinhdv/github/yolov5_0509/runs/MILKa/crops/Sua/0aee8754d1.jpg')
    augmenter = RandAugment(list(AUGMENT_FUNCTION.keys()))
    for _ in range(30):
        img_aug = augmenter(img_org)
        img_pad = preprocess(img_aug,224)
        cv2.imshow('a',img_org)
        cv2.imshow('b',img_aug)
        cv2.imshow('c',img_pad)
        if cv2.waitKey(0)==ord('q'):
            exit()
