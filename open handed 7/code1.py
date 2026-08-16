import glob,os
import cv2,numpy as np
import matplotlib.pyplot as plt

BASE_DIR=r"D:\sem-5\dsip\open handed 7"
IMAGE_DIR=BASE_DIR

files=sorted(glob.glob(os.path.join(IMAGE_DIR,"*.jpg"))+glob.glob(os.path.join(IMAGE_DIR,"*.jpeg"))+glob.glob(os.path.join(IMAGE_DIR,"*.png")))[:5]

images=[cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in files]

def contrast_stretch(img):
    min_val,max_val=np.min(img),np.max(img)
    if max_val==min_val:
        return img
    return np.uint8((img.astype(np.float32)-min_val)*255.0/(max_val-min_val))

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def gamma_correction(img,gamma=2.2):
    norm=img/255.0
    return np.uint8(np.power(norm,gamma)*255.0)

def log_transform(img):
    norm=img.astype(np.float32)
    c=255.0/np.log(1.0+np.max(norm))
    return np.uint8(c*np.log1p(norm))

def image_negative(img):
    return 255-img

methods=[
("Contrast Stretching",contrast_stretch),
("Histogram Equalization",histogram_equalization),
("Gamma Transformation (γ=2.2)",gamma_correction),
("Log Transformation",log_transform),
("Image Negative",image_negative)
]

enhanced_images=[]

for i,(name,func) in enumerate(methods):
    enh=func(images[i])
    enhanced_images.append(enh)

fig,axes=plt.subplots(5,2,figsize=(10,18))

for i in range(5):
    name,_=methods[i]
    axes[i,0].imshow(images[i],cmap="gray")
    axes[i,0].set_title(f"Image {i+1} - Original")
    axes[i,0].axis("off")
    axes[i,1].imshow(enhanced_images[i],cmap="gray")
    axes[i,1].set_title(f"Image {i+1} - {name}")
    axes[i,1].axis("off")

plt.tight_layout()
plt.show()

fig_hist,axes_hist=plt.subplots(5,2,figsize=(12,16))

for i in range(5):
    name,_=methods[i]
    axes_hist[i,0].hist(images[i].ravel(),bins=256,range=[0,256])
    axes_hist[i,0].set_title(f"Image {i+1} Original Histogram")
    axes_hist[i,0].set_xlim([0,256])
    axes_hist[i,1].hist(enhanced_images[i].ravel(),bins=256,range=[0,256])
    axes_hist[i,1].set_title(f"Image {i+1} Enhanced Histogram ({name})")
    axes_hist[i,1].set_xlim([0,256])

plt.tight_layout()
plt.show()