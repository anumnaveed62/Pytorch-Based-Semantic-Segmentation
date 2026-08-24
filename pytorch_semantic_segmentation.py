
from torchvision import models
from PIL import Image

import torchvision
import os
import subprocess
import matplotlib.pyplot as plt
import torch
import requests
import time

plt.style.use('ggplot')

# Load the FCN ResNet101 segmentation model.
fcn = models.segmentation.fcn_resnet101(
    weights=torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
).eval()

"""And that's it we have a pretrained model of `FCN` (which stands for Fully Convolutional Neural Networks) with a `Resnet101` backbone :)

Now, let's get an image!
"""

# Helper function to download image and other files.
# Helper function to download file.
def download_file(url, save_name):
    if not os.path.exists(save_name):
        subprocess.run(['wget', url, '-O', save_name, '-q'])

# Create directory to store inference data
inference_dir = 'inference_data'
os.makedirs(inference_dir, exist_ok=True)

from google.colab import drive
drive.mount('/content/drive')

download_file(
    'https://learnopencv.com/wp-content/uploads/2022/10/bird.jpg',
    save_name=os.path.join(inference_dir, 'bird.jpg')
)
img = Image.open(os.path.join(inference_dir, 'bird.jpg'))
plt.imshow(img)
plt.axis('off')
plt.show()

"""Now, that we have the image we need to preprocess it and normalize it! <br/>
So, for the preprocessing steps, we:
- Resize the image to `(256 x 256)`
- CenterCrop it to `(224 x 224)`
- Convert it to Tensor - all the values in the image becomes between `[0, 1]` from `[0, 255]`
- Normalize it with the Imagenet specific values `mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]`

And lastly, we unsqueeze the image so that it becomes `[1 x C x H x W]` from `[C x H x W]` <br/>
We need a batch dimension while passing it to the models.
"""

# Apply the transformations needed
import torchvision.transforms as T
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)

"""Let's see what the above code cell does </br>
`T.Compose` is a function that takes in a `list` in which each element is of `transforms` type and </br>
it returns a object through which we can
pass batches of images and all the required transforms will be applied to the images.

Let's take a look at the transforms applied on the images:
- `T.Resize(256)` : Resizes the image to size `256 x 256`
- `T.CenterCrop(224)` : Center Crops the image to have a resulting size of `224 x 224`
- `T.ToTensor()` : Converts the image to type `torch.Tensor` and have values between `[0, 1]`
- `T.Normalize(mean, std)` : Normalizes the image with the given mean and standard deviation.

Alright! Now that we have the image all preprocessed and ready! Let's pass it through the model and get the `out` key.<br/>
As I said, the output of the model is a `OrderedDict` so, we need to take the `out` key from that to get the output of the model.
"""

# Pass the input through the net
out = fcn(inp)['out']
print (out.shape)

"""Alright! So, `out` is the final output of the model. And as we can see, its shape is `[1 x 21 x H x W]` as discussed earlier. So, the model was trained on `21` classes and thus our output have `21` channels!<br/>

Now, what we need to do is make this `21` channeled output into a `2D` image or a `1` channeled image, where each pixel of that image corresponds to a class!

So, the `2D` image, (of shape `[H x W]`) will have each pixel corresponding to a class label, and thus <br/>
for each `(x, y)` in this `2D` image will correspond to a number between `0 - 20` representing a class.

And how do we get there from this `[1 x 21 x H x W]`?<br/>
We take a max index for each pixel position, which represents the class<br/>
"""

import numpy as np
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print (om.shape)
print (np.unique(om))

"""Alright! So, we as we can see now have a `2D` image. Where each pixel corresponds to a class!
The last thing is to take this `2D` image where each pixel corresponds to a class label and convert this<br/>
into a segmentation map where each class label is converted into a `RGB` color and thus helping in easy visualization.

We will use the following function to convert this `2D` image to an `RGB` image wheree each label is mapped to its
corresponding color.
"""

# Define the helper function
def decode_segmap(image, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

"""Let's see what we are doing inside this function!

first `label_colors` stores the colors for each of the clases, according to the index </br>
So, the color for the  first class which is `background` is stored in the `0`th index of the `label_colors` list,
the second class which is `aeroplane` is stored at index `1` of `label_colors`.

Now, we are to create an `RGB` image from the `2D` image passed. So, what we do, is we create empty `2D` matrices for all 3 channels.

So, `r`, `g`, and `b` are arrays which will form the `RGB` channels for the final image. And each are of shape `[H x W]`
(which is same as the shape of `image` passed in)

Now, we loop over each class color we stored in `label_colors`.
And we get the indexes in the image where that particular class label is present. (`idx = image == l`)
And then for each channel, we put its corresponding color to those pixels where that class label is present.

And finally we stack the 3 seperate channels to form a `RGB` image.

Okay! Now, let's use this function to see the final segmented output!
"""

rgb = decode_segmap(om)
plt.figure(figsize=(12, 9))
plt.imshow(rgb)
plt.axis('off')
plt.show()

"""And there we go!!<br/>
Wooohooo! We have segmented the output of the image.

That's the bird!

Also, Do note that the image after segmentation is smaller than the original image as in the preprocessing step the image is resized and cropped.

Next, let's move all this under one function and play with a few more images!
"""

def segment(net, path, show_orig=True, dev='cuda'):
    plt.figure(figsize=(10, 7))
    img = Image.open(path)
    if show_orig:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([T.Resize(640),
                   #T.CenterCrop(224),
                   T.ToTensor(),
                   T.Normalize(mean = [0.485, 0.456, 0.406],
                               std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    plt.figure(figsize=(10, 7))
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

"""And let's get a new image!"""

download_file(
    'https://www.learnopencv.com/wp-content/uploads/2021/01/horse-segmentation.jpeg',
     save_name=os.path.join(inference_dir, 'horse.jpg')
)
segment(fcn, os.path.join(inference_dir, 'horse.jpg'))

"""### DeepLabv3"""

dlab = models.segmentation.deeplabv3_resnet101(
    weights='COCO_WITH_VOC_LABELS_V1'
#     pretrained=True
).eval()

"""Alright! Now we have god-level segmentation model!<br/>
Let's see how we perform with the same image on this model!
"""

segment(dlab, os.path.join(inference_dir, 'horse.jpg'))

"""Yeah! So, there you go! You can see that, the DeepLab model has also classified the image quite nicely! </br>
But if we take a more complex image! Then we start to see model differences!

Note: As we saw before the output image size is smaller than the original image as the original image is resized and cropped in the preprocessing step.

Let's try that out!
"""

download_file(
    'https://www.learnopencv.com/wp-content/uploads/2021/01/person-segmentation.jpeg',
    save_name=os.path.join(inference_dir, 'person.jpg')
)
img = Image.open(os.path.join(inference_dir, 'person.jpg'))
plt.imshow(img)
plt.axis('off')
plt.show()

print ('Segmenatation Image on FCN')
segment(fcn, path=os.path.join(inference_dir, 'person.jpg'), show_orig=False)

print ('Segmenatation Image on DeepLabv3')
segment(dlab, path=os.path.join(inference_dir, 'person.jpg'), show_orig=False)

"""Okay! You can now see the model differences right?

You can see how FCN fails to capture the continuity of the leg of the cow while DeepLabv3 is able to capture that!

Also, if we look closer into the hand of the human which is on the cow, we can see that the FCN model captures it nicely, not very nicely, but still, while the DeepLabv3 model has captured it too but not that well!

These are a few model differences that be noticed with bare eyes!

Note: As we saw before the output image size is smaller than the original image as the original image is resized and cropped in the preprocessing step.

Do play around with a few more images to see how these models perform in different scenarios.!

## Comparison

For, now we will see how these two models compare with each other in 3 metrics
- Inference time
- Size of the model
- GPU memory used by the model

### Inference Time
"""

def infer_time(net, path=os.path.join(inference_dir, 'horse.jpg'), dev='cuda'):
    img = Image.open(path)
    trf = T.Compose([T.Resize(256),
                   T.CenterCrop(224),
                   T.ToTensor(),
                   T.Normalize(mean = [0.485, 0.456, 0.406],
                               std = [0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0).to(dev)

    st = time.time()
    out1 = net.to(dev)(inp)
    et = time.time()

    return et - st

"""**On CPU**"""

avg_over = 100

fcn_infer_time_list_cpu = [infer_time(fcn, dev='cpu') for _ in range(avg_over)]
fcn_infer_time_avg_cpu = sum(fcn_infer_time_list_cpu) / avg_over

dlab_infer_time_list_cpu = [infer_time(dlab, dev='cpu') for _ in range(avg_over)]
dlab_infer_time_avg_cpu = sum(dlab_infer_time_list_cpu) / avg_over


print ('Inference time for first few calls for FCN      : {}'.format(fcn_infer_time_list_cpu[:10]))
print ('Inference time for first few calls for DeepLabv3: {}'.format(dlab_infer_time_list_cpu[:10]))

print ('The Average Inference time on FCN is:     {:.2f}s'.format(fcn_infer_time_avg_cpu))
print ('The Average Inference time on DeepLab is: {:.2f}s'.format(dlab_infer_time_avg_cpu))

"""**On GPU**"""

avg_over = 100

fcn_infer_time_list_gpu = [infer_time(fcn) for _ in range(avg_over)]
fcn_infer_time_avg_gpu = sum(fcn_infer_time_list_gpu) / avg_over

dlab_infer_time_list_gpu = [infer_time(dlab) for _ in range(avg_over)]
dlab_infer_time_avg_gpu = sum(dlab_infer_time_list_gpu) / avg_over

print ('Inference time for first few calls for FCN      : {}'.format(fcn_infer_time_list_gpu[:10]))
print ('Inference time for first few calls for DeepLabv3: {}'.format(dlab_infer_time_list_gpu[:10]))

print ('The Average Inference time on FCN is:     {:.3f}s'.format(fcn_infer_time_avg_gpu))
print ('The Average Inference time on DeepLab is: {:.3f}s'.format(dlab_infer_time_avg_gpu))

"""
We can see that in both cases (for GPU and CPU) the inference time of DeepLabv3 and FCN is about the same even though DeepLabv3 is theoretically speaking a computationally more expensive model.

Also, we have printed out the first few inference times for each model. Something we can notice is that the inference time for the first call
takes quite long than the others . This is because after the 1st call a lot of the calculations required are cached and thus its faster for the next calls.

Nice! Now, let's try to vizualize the difference in the time taken for the CPU and the GPU."""

plt.figure(figsize=(10, 7))
plt.bar([0.1, 0.2], [fcn_infer_time_avg_cpu, dlab_infer_time_avg_cpu], width=0.08)
plt.ylabel('Time taken in Seconds')
plt.xticks([0.1, 0.2], ['FCN', 'DeepLabv3'])
plt.title('Inference time of FCN and DeepLabv3 on CPU')
plt.show()

plt.figure(figsize=(10, 7))
plt.bar([0.1, 0.2], [fcn_infer_time_avg_gpu, dlab_infer_time_avg_gpu], width=0.08)
plt.ylabel('Time taken in Seconds')
plt.xticks([0.1, 0.2], ['FCN', 'DeepLabv3'])
plt.title('Inference time of FCN and DeepLabv3 on GPU')
plt.show()

"""## Conclusion

Hope you enjoyed this tutorial!

Feel free to leave comments and any feedback you wish! If you would like to learn<br/>
more about this, how these techniques work and how to implement these models!

Please do check out the `Deep Learning with PyTorch` Course from OpenCV.org! <br/>
Link: [https://opencv.org/university/deep-learning-with-pytorch/](https://opencv.org/university/deep-learning-with-pytorch/)
"""

# =============================================
# 3-CLASS SEGMENTATION: Person + Red Sunglasses
# =============================================

import torch
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# -------------------------------
# 1. 3-Class Decode Function
# -------------------------------
def decode_3class(image):
    colors = np.array([
        [0,   0,   0],    # 0: Background → Black
        [192, 128, 128],  # 1: Person     → Light Purple
        [255, 0,   0]     # 2: Sunglasses → Bright Red
    ], dtype=np.uint8)

    h, w = image.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        rgb[image == i] = colors[i]
    return rgb

# -------------------------------
# 2. Load Image
# -------------------------------
image_path = '/content/drive/MyDrive/rushina-morrison-7OWzhjXNwWQ-unsplash.jpg'  # Change to your path
img_pil = Image.open(image_path).convert('RGB')
img_np = np.array(img_pil)

plt.figure(figsize=(8, 10))
plt.imshow(img_np)
plt.axis('off')
plt.title('Original Image')
plt.show()

# -------------------------------
# 3. Preprocess
# -------------------------------
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img_pil).unsqueeze(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 4. Load DeepLabV3 (VOC pretrained)
# -------------------------------
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()

with torch.no_grad():
    output = model(input_tensor.to(device))['out'][0]  # [21, 224, 224]
    pred = output.argmax(0).cpu().numpy()  # (224, 224)

# -------------------------------
# 5. Base 3-class map: BG=0, Person=1
# -------------------------------
seg = np.zeros_like(pred, dtype=np.uint8)
seg[pred == 0] = 0   # background
seg[pred == 15] = 1  # person (VOC class 15)

# -------------------------------
# 6. Detect RED SUNGLASSES using color thresholding
# -------------------------------
# Convert original image to HSV for robust red detection
img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

# Define red color range (two ranges: 0-10 and 170-180)
lower_red1 = np.array([0, 70, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 70])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
red_mask = mask1 | mask2  # Combine

# Resize red mask to model size (224x224)
red_mask_resized = cv2.resize(red_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
red_mask_resized = red_mask_resized.astype(bool)

# Refine: only keep red where person is predicted
sunglasses_mask = red_mask_resized & (seg == 1)

# Assign class 2 to sunglasses
seg[sunglasses_mask] = 2

# -------------------------------
# 7. Decode to RGB
# -------------------------------
seg_rgb = decode_3class(seg)

plt.figure(figsize=(8, 10))
plt.imshow(seg_rgb)
plt.axis('off')
plt.title('3-Class Segmentation\nBlack=BG, Purple=Person, Red=Sunglasses')
plt.show()

# -------------------------------
# 8. Overlay on Original
# -------------------------------
def overlay_seg(orig, mask, alpha=0.6):
    orig = np.array(orig)
    mask_resized = Image.fromarray(mask).resize(orig.shape[1::-1], Image.NEAREST)
    mask_np = np.array(mask_resized)

    overlay = orig.copy()
    active = mask_np.sum(axis=2) > 0
    overlay[active] = (alpha * orig[active] + (1 - alpha) * mask_np[active]).astype(np.uint8)
    return overlay

overlay = overlay_seg(img_pil, seg_rgb, alpha=0.6)

plt.figure(figsize=(8, 10))
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay: Person (Purple) + Sunglasses (Red)')
plt.show()

# =============================================
# 3-CLASS SEGMENTATION: Person (Red) + Sunglasses (Orange)
# =============================================

import torch
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# -------------------------------
# 1. 3-Class Decode Function (Red, Orange, Black)
# -------------------------------
def decode_3class(image):
    colors = np.array([
        [0,   0,   0],    # 0: Background → Black
        [255, 0,   0],    # 1: Person → Red
        [255, 165, 0]     # 2: Sunglasses → Orange
    ], dtype=np.uint8)

    h, w = image.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        rgb[image == i] = colors[i]
    return rgb

# -------------------------------
# 2. Load Image
# -------------------------------
image_path = '/content/drive/MyDrive/rushina-morrison-7OWzhjXNwWQ-unsplash.jpg'  # Update if needed
img_pil = Image.open(image_path).convert('RGB')
img_np = np.array(img_pil)

plt.figure(figsize=(8, 10))
plt.imshow(img_np)
plt.axis('off')
plt.title('Original Image')
plt.show()

# -------------------------------
# 3. Preprocess
# -------------------------------
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img_pil).unsqueeze(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 4. Load DeepLabV3 (VOC pretrained)
# -------------------------------
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()

with torch.no_grad():
    output = model(input_tensor.to(device))['out'][0]  # [21, 224, 224]
    pred = output.argmax(0).cpu().numpy()  # (224, 224)

# -------------------------------
# 5. Base 3-class: BG=0, Person=1
# -------------------------------
seg = np.zeros_like(pred, dtype=np.uint8)
seg[pred == 0] = 0   # background
seg[pred == 15] = 1  # person → will be red

# -------------------------------
# 6. Detect RED/ORANGE SUNGLASSES using HSV
# -------------------------------
img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

# Red + Orange range in HSV
lower_red_orange1 = np.array([0, 70, 70])      # Red low
upper_red_orange1 = np.array([10, 255, 255])
lower_red_orange2 = np.array([160, 70, 70])   # Red high
upper_red_orange2 = np.array([180, 255, 255])
lower_orange = np.array([11, 70, 70])         # Orange
upper_orange = np.array([25, 255, 255])

mask_red1 = cv2.inRange(img_hsv, lower_red_orange1, upper_red_orange1)
mask_red2 = cv2.inRange(img_hsv, lower_red_orange2, upper_red_orange2)
mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)

red_orange_mask = mask_red1 | mask_red2 | mask_orange

# Resize to model size
red_orange_mask_resized = cv2.resize(red_orange_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
red_orange_mask_resized = red_orange_mask_resized.astype(bool)

# Refine: only on person's face (upper part)
face_region = seg[20:80, 60:160] == 1  # approximate face area in 224x224
face_mask = np.zeros_like(seg, dtype=bool)
face_mask[20:80, 60:160] = face_region

sunglasses_mask = red_orange_mask_resized & face_mask

# Optional: clean small noise
kernel = np.ones((3,3), np.uint8)
sunglasses_mask = cv2.morphologyEx(sunglasses_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
sunglasses_mask = sunglasses_mask.astype(bool)

# Assign class 2 (orange)
seg[sunglasses_mask] = 2

# -------------------------------
# 7. Decode to RGB (Red, Orange, Black)
# -------------------------------
seg_rgb = decode_3class(seg)

plt.figure(figsize=(8, 10))
plt.imshow(seg_rgb)
plt.axis('off')
plt.title('Segmentation: Red=Person, Orange=Sunglasses, Black=BG')
plt.show()

# -------------------------------
# 8. Overlay on Original
# -------------------------------
def overlay_seg(orig, mask, alpha=0.6):
    orig = np.array(orig)
    mask_resized = Image.fromarray(mask).resize(orig.shape[1::-1], Image.NEAREST)
    mask_np = np.array(mask_resized)

    overlay = orig.copy()
    active = mask_np.sum(axis=2) > 0
    overlay[active] = (alpha * orig[active] + (1 - alpha) * mask_np[active]).astype(np.uint8)
    return overlay

overlay = overlay_seg(img_pil, seg_rgb, alpha=0.6)

plt.figure(figsize=(8, 10))
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay: Red=Person, Orange=Sunglasses')
plt.show()

# =============================================
# 3-CLASS SEGMENTATION: Girl + Yellow Bicycle
# =============================================

# Import required libraries
import torch  # PyTorch for deep learning model inference
import torchvision.transforms as T  # Image preprocessing transforms
from torchvision import models  # Pretrained segmentation models
import matplotlib.pyplot as plt  # For displaying images and results
import numpy as np  # Numerical operations on arrays
from PIL import Image  # Load and manipulate images
import cv2  # OpenCV for color-based masking (HSV)

# -------------------------------
# 1. 3-Class Decode Function
# -------------------------------
def decode_3class(image):
    """
    Converts a label map (0, 1, 2) into an RGB image using custom colors.
    Input: image - (H, W) numpy array with integer labels
    Output: RGB image (H, W, 3) with uint8 values
    """
    # Define RGB colors for each class
    colors = np.array([
        [0,   0,   0],    # Class 0: Background → Black
        [192, 128, 128],  # Class 1: Girl (Person) → Light Purple
        [255, 255, 0]     # Class 2: Bicycle → Yellow
    ], dtype=np.uint8)

    h, w = image.shape  # Get height and width of the label map
    rgb = np.zeros((h, w, 3), dtype=np.uint8)  # Initialize empty RGB image

    # Loop through each class (0, 1, 2)
    for i in range(3):
        # Create a boolean mask where label == i
        rgb[image == i] = colors[i]  # Assign corresponding color to all pixels with label i

    return rgb  # Return the colored segmentation map

# -------------------------------
# 2. Load Image
# -------------------------------
image_path = '/content/drive/MyDrive/Girl-with-cycle.jpg'  # Path to input image (update if needed)
img_pil = Image.open(image_path).convert('RGB')  # Open image and ensure it's in RGB format
img_np = np.array(img_pil)  # Convert PIL image to NumPy array for OpenCV processing

# Display original image
plt.figure(figsize=(8, 12))  # Set figure size
plt.imshow(img_np)  # Show the image
plt.axis('off')  # Hide axes
plt.title('Original Image')  # Add title
plt.show()  # Render the plot

# -------------------------------
# 3. Preprocess
# -------------------------------
# Define image transformations to match DeepLabV3 input requirements
transform = T.Compose([
    T.Resize(256),  # Resize shortest side to 256 while keeping aspect ratio
    T.CenterCrop(224),  # Crop center 224x224 region (standard input size)
    T.ToTensor(),  # Convert PIL image to PyTorch tensor (C, H, W) and normalize to [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

input_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# -------------------------------
# 4. Load DeepLabV3 (VOC pretrained)
# -------------------------------
# Load DeepLabV3 with ResNet50 backbone, pretrained on PASCAL VOC (21 classes)
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()  # Set model to evaluation mode (no dropout, batchnorm frozen)

# Run inference without gradient computation
with torch.no_grad():
    output = model(input_tensor.to(device))['out'][0]  # Forward pass: get logits [21, 224, 224]
    pred = output.argmax(0).cpu().numpy()  # Get predicted class per pixel: [224, 224]

# -------------------------------
# 5. Base 3-class: BG=0, Girl=1
# -------------------------------
seg = np.zeros_like(pred, dtype=np.uint8)  # Initialize segmentation map with zeros
seg[pred == 0] = 0   # All background pixels (VOC class 0) → label 0
seg[pred == 15] = 1  # All person pixels (VOC class 15) → label 1 (will be purple)

# -------------------------------
# 6. Detect YELLOW BICYCLE using HSV
# -------------------------------
img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)  # Convert RGB image to HSV color space

# Define yellow color range in HSV:
# Hue: 20–40 (yellow range), Saturation > 70, Value > 70
lower_yellow = np.array([20, 70, 70])
upper_yellow = np.array([40, 255, 255])

# Create binary mask where pixels fall in yellow range
yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

# Resize mask from original image size to model size (224x224)
yellow_mask_resized = cv2.resize(yellow_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
yellow_mask_resized = yellow_mask_resized.astype(bool)  # Convert to boolean mask

# Optional: Remove small noise using morphological opening
kernel = np.ones((3,3), np.uint8)  # 3x3 structuring element
yellow_mask_clean = cv2.morphologyEx(yellow_mask_resized.astype(np.uint8), cv2.MORPH_OPEN, kernel)
yellow_mask_clean = yellow_mask_clean.astype(bool)  # Cleaned binary mask

# Final bicycle mask: yellow AND not part of person (avoid overlap on hands/clothes)
bicycle_mask = yellow_mask_clean & (seg != 1)
seg[bicycle_mask] = 2  # Assign label 2 to bicycle pixels (will be yellow)

# -------------------------------
# 7. Decode to RGB
# -------------------------------
seg_rgb = decode_3class(seg)  # Convert label map to colored RGB image

# Display segmentation result
plt.figure(figsize=(8, 12))
plt.imshow(seg_rgb)
plt.axis('off')
plt.title('3-Class Segmentation\nBlack=BG, Purple=Girl, Yellow=Bicycle')
plt.show()

# -------------------------------
# 8. Overlay on Original
# -------------------------------
def overlay_seg(orig, mask, alpha=0.6):
    """
    Overlays segmentation mask on original image with transparency.
    orig: original RGB image (H, W, 3)
    mask: colored segmentation map (H, W, 3)
    alpha: transparency of overlay (0=original, 1=mask)
    """
    orig = np.array(orig)  # Convert PIL to NumPy
    # Resize mask to original image size using nearest neighbor (preserves sharp edges)
    mask_resized = Image.fromarray(mask).resize(orig.shape[1::-1], Image.NEAREST)
    mask_np = np.array(mask_resized)

    overlay = orig.copy()  # Start with original image
    active = mask_np.sum(axis=2) > 0  # Boolean mask: where segmentation exists
    # Blend: alpha * original + (1-alpha) * mask
    overlay[active] = (alpha * orig[active] + (1 - alpha) * mask_np[active]).astype(np.uint8)
    return overlay

# Apply overlay
overlay = overlay_seg(img_pil, seg_rgb, alpha=0.6)

# Display final result
plt.figure(figsize=(8, 12))
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay: Girl (Purple) + Bicycle (Yellow)')
plt.show()

# =============================================
# 3-CLASS SEGMENTATION: Person (Blue) + Sunglasses (Green)
# =============================================

import torch
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# -------------------------------
# 1. 3-Class Decode Function (Blue, Green, Black)
# -------------------------------
def decode_3class(image):
    colors = np.array([
        [0,   0,   0],    # 0: Background → Black
        [0,   0, 255],    # 1: Person → Blue
        [0, 255,   0]     # 2: Sunglasses → Green
    ], dtype=np.uint8)

    h, w = image.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        rgb[image == i] = colors[i]
    return rgb

# -------------------------------
# 2. Load Image
# -------------------------------
image_path = '/content/drive/MyDrive/rushina-morrison-7OWzhjXNwWQ-unsplash.jpg'  # Update if needed
img_pil = Image.open(image_path).convert('RGB')
img_np = np.array(img_pil)

plt.figure(figsize=(8, 10))
plt.imshow(img_np)
plt.axis('off')
plt.title('Original Image')
plt.show()

# -------------------------------
# 3. Preprocess
# -------------------------------
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img_pil).unsqueeze(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 4. Load DeepLabV3 (VOC pretrained)
# -------------------------------
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()

with torch.no_grad():
    output = model(input_tensor.to(device))['out'][0]  # [21, 224, 224]
    pred = output.argmax(0).cpu().numpy()  # (224, 224)

# -------------------------------
# 5. Base 3-class: BG=0, Person=1
# -------------------------------
seg = np.zeros_like(pred, dtype=np.uint8)
seg[pred == 0] = 0   # background
seg[pred == 15] = 1  # person → will be blue

# -------------------------------
# 6. Detect RED SUNGLASSES using HSV (still detect red, but color it green)
# -------------------------------
img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

# Red color range (two ranges)
lower_red1 = np.array([0, 70, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 70])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
red_mask = mask1 | mask2

# Resize to model size
red_mask_resized = cv2.resize(red_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
red_mask_resized = red_mask_resized.astype(bool)

# Refine: only on person's face region
face_region = (seg[20:80, 60:160] == 1)
face_mask = np.zeros_like(seg, dtype=bool)
face_mask[20:80, 60:160] = face_region

sunglasses_mask = red_mask_resized & face_mask

# Clean small noise
kernel = np.ones((3,3), np.uint8)
sunglasses_mask = cv2.morphologyEx(sunglasses_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
sunglasses_mask = sunglasses_mask.astype(bool)

# Assign class 2 → will be colored green
seg[sunglasses_mask] = 2

# -------------------------------
# 7. Decode to RGB (Blue, Green, Black)
# -------------------------------
seg_rgb = decode_3class(seg)

plt.figure(figsize=(8, 10))
plt.imshow(seg_rgb)
plt.axis('off')
plt.title('Segmentation: Blue=Person, Green=Sunglasses, Black=Background')
plt.show()

# -------------------------------
# 8. Overlay on Original
# -------------------------------
def overlay_seg(orig, mask, alpha=0.6):
    orig = np.array(orig)
    mask_resized = Image.fromarray(mask).resize(orig.shape[1::-1], Image.NEAREST)
    mask_np = np.array(mask_resized)

    overlay = orig.copy()
    active = mask_np.sum(axis=2) > 0
    overlay[active] = (alpha * orig[active] + (1 - alpha) * mask_np[active]).astype(np.uint8)
    return overlay

overlay = overlay_seg(img_pil, seg_rgb, alpha=0.6)

plt.figure(figsize=(8, 10))
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay: Blue=Person, Green=Sunglasses')
plt.show()

# =============================================
# 3-CLASS SEGMENTATION: Girl + Yellow Bicycle
# =============================================

import torch
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# -------------------------------
# 1. 3-Class Decode Function
# -------------------------------
def decode_3class(image):
    colors = np.array([
        [0,   0,   0],    # 0: Background → Black
        [192, 128, 128],  # 1: Girl (Person) → Light Purple
        [255, 255, 0]     # 2: Bicycle → Yellow
    ], dtype=np.uint8)

    h, w = image.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        rgb[image == i] = colors[i]
    return rgb

# -------------------------------
# 2. Load Image
# -------------------------------
image_path = '/content/drive/MyDrive/Girl-with-cycle.jpg'  # Change to your path
img_pil = Image.open(image_path).convert('RGB')
img_np = np.array(img_pil)

plt.figure(figsize=(8, 12))
plt.imshow(img_np)
plt.axis('off')
plt.title('Original Image')
plt.show()

# -------------------------------
# 3. Preprocess
# -------------------------------
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img_pil).unsqueeze(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 4. Load DeepLabV3 (VOC pretrained)
# -------------------------------
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()

with torch.no_grad():
    output = model(input_tensor.to(device))['out'][0]  # [21, 224, 224]
    pred = output.argmax(0).cpu().numpy()  # (224, 224)

# -------------------------------
# 5. Base 3-class: BG=0, Girl=1
# -------------------------------
seg = np.zeros_like(pred, dtype=np.uint8)
seg[pred == 0] = 0   # background
seg[pred == 15] = 1  # person (VOC class 15)

# -------------------------------
# 6. Detect YELLOW BICYCLE using HSV
# -------------------------------
img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

# Yellow range in HSV: Hue 25-35, Sat > 70, Val > 70
lower_yellow = np.array([20, 70, 70])
upper_yellow = np.array([40, 255, 255])

yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

# Resize to model size
yellow_mask_resized = cv2.resize(yellow_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
yellow_mask_resized = yellow_mask_resized.astype(bool)

# Optional: Remove small noise
kernel = np.ones((3,3), np.uint8)
yellow_mask_clean = cv2.morphologyEx(yellow_mask_resized.astype(np.uint8), cv2.MORPH_OPEN, kernel)
yellow_mask_clean = yellow_mask_clean.astype(bool)

# Assign bicycle only where not person (avoid overlap)
bicycle_mask = yellow_mask_clean & (seg != 1)
seg[bicycle_mask] = 2

# -------------------------------
# 7. Decode to RGB
# -------------------------------
seg_rgb = decode_3class(seg)

plt.figure(figsize=(8, 12))
plt.imshow(seg_rgb)
plt.axis('off')
plt.title('3-Class Segmentation\nBlack=BG, Purple=Girl, Yellow=Bicycle')
plt.show()

# -------------------------------
# 8. Overlay on Original
# -------------------------------
def overlay_seg(orig, mask, alpha=0.6):
    orig = np.array(orig)
    mask_resized = Image.fromarray(mask).resize(orig.shape[1::-1], Image.NEAREST)
    mask_np = np.array(mask_resized)

    overlay = orig.copy()
    active = mask_np.sum(axis=2) > 0
    overlay[active] = (alpha * orig[active] + (1 - alpha) * mask_np[active]).astype(np.uint8)
    return overlay

overlay = overlay_seg(img_pil, seg_rgb, alpha=0.6)

plt.figure(figsize=(8, 12))
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay: Girl (Purple) + Bicycle (Yellow)')
plt.show()

# =============================================
# 3-CLASS SEGMENTATION: Girl + Yellow Bicycle
# =============================================

# Import required libraries
import torch  # PyTorch for deep learning model inference
import torchvision.transforms as T  # Image preprocessing transforms
from torchvision import models  # Pretrained segmentation models
import matplotlib.pyplot as plt  # For displaying images and results
import numpy as np  # Numerical operations on arrays
from PIL import Image  # Load and manipulate images
import cv2  # OpenCV for color-based masking (HSV)

# -------------------------------
# 1. 3-Class Decode Function
# -------------------------------
def decode_3class(image):
    """
    Converts a label map (0, 1, 2) into an RGB image using custom colors.
    Input: image - (H, W) numpy array with integer labels
    Output: RGB image (H, W, 3) with uint8 values
    """
    # Define RGB colors for each class
    colors = np.array([
        [0,   0,   0],    # Class 0: Background → Black
        [192, 128, 128],  # Class 1: Girl (Person) → Light Purple
        [255, 255, 0]     # Class 2: Bicycle → Yellow
    ], dtype=np.uint8)

    h, w = image.shape  # Get height and width of the label map
    rgb = np.zeros((h, w, 3), dtype=np.uint8)  # Initialize empty RGB image

    # Loop through each class (0, 1, 2)
    for i in range(3):
        # Create a boolean mask where label == i
        rgb[image == i] = colors[i]  # Assign corresponding color to all pixels with label i

    return rgb  # Return the colored segmentation map

# -------------------------------
# 2. Load Image
# -------------------------------
image_path = '/content/drive/MyDrive/Girl-with-cycle.jpg'  # Path to input image (update if needed)
img_pil = Image.open(image_path).convert('RGB')  # Open image and ensure it's in RGB format
img_np = np.array(img_pil)  # Convert PIL image to NumPy array for OpenCV processing

# Display original image
plt.figure(figsize=(8, 12))  # Set figure size
plt.imshow(img_np)  # Show the image
plt.axis('off')  # Hide axes
plt.title('Original Image')  # Add title
plt.show()  # Render the plot

# -------------------------------
# 3. Preprocess
# -------------------------------
# Define image transformations to match DeepLabV3 input requirements
transform = T.Compose([
    T.Resize(256),  # Resize shortest side to 256 while keeping aspect ratio
    T.CenterCrop(224),  # Crop center 224x224 region (standard input size)
    T.ToTensor(),  # Convert PIL image to PyTorch tensor (C, H, W) and normalize to [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

input_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# -------------------------------
# 4. Load DeepLabV3 (VOC pretrained)
# -------------------------------
# Load DeepLabV3 with ResNet50 backbone, pretrained on PASCAL VOC (21 classes)
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()  # Set model to evaluation mode (no dropout, batchnorm frozen)

# Run inference without gradient computation
with torch.no_grad():
    output = model(input_tensor.to(device))['out'][0]  # Forward pass: get logits [21, 224, 224]
    pred = output.argmax(0).cpu().numpy()  # Get predicted class per pixel: [224, 224]

# -------------------------------
# 5. Base 3-class: BG=0, Girl=1
# -------------------------------
seg = np.zeros_like(pred, dtype=np.uint8)  # Initialize segmentation map with zeros
seg[pred == 0] = 0   # All background pixels (VOC class 0) → label 0
seg[pred == 15] = 1  # All person pixels (VOC class 15) → label 1 (will be purple)

# -------------------------------
# 6. Detect YELLOW BICYCLE using HSV
# -------------------------------
img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)  # Convert RGB image to HSV color space

# Define yellow color range in HSV:
# Hue: 20–40 (yellow range), Saturation > 70, Value > 70
lower_yellow = np.array([20, 70, 70])
upper_yellow = np.array([40, 255, 255])

# Create binary mask where pixels fall in yellow range
yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

# Resize mask from original image size to model size (224x224)
yellow_mask_resized = cv2.resize(yellow_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
yellow_mask_resized = yellow_mask_resized.astype(bool)  # Convert to boolean mask

# Optional: Remove small noise using morphological opening
kernel = np.ones((3,3), np.uint8)  # 3x3 structuring element
yellow_mask_clean = cv2.morphologyEx(yellow_mask_resized.astype(np.uint8), cv2.MORPH_OPEN, kernel)
yellow_mask_clean = yellow_mask_clean.astype(bool)  # Cleaned binary mask

# Final bicycle mask: yellow AND not part of person (avoid overlap on hands/clothes)
bicycle_mask = yellow_mask_clean & (seg != 1)
seg[bicycle_mask] = 2  # Assign label 2 to bicycle pixels (will be yellow)

# -------------------------------
# 7. Decode to RGB
# -------------------------------
seg_rgb = decode_3class(seg)  # Convert label map to colored RGB image

# Display segmentation result
plt.figure(figsize=(8, 12))
plt.imshow(seg_rgb)
plt.axis('off')
plt.title('3-Class Segmentation\nBlack=BG, Purple=Girl, Yellow=Bicycle')
plt.show()

# -------------------------------
# 8. Overlay on Original
# -------------------------------
def overlay_seg(orig, mask, alpha=0.6):
    """
    Overlays segmentation mask on original image with transparency.
    orig: original RGB image (H, W, 3)
    mask: colored segmentation map (H, W, 3)
    alpha: transparency of overlay (0=original, 1=mask)
    """
    orig = np.array(orig)  # Convert PIL to NumPy
    # Resize mask to original image size using nearest neighbor (preserves sharp edges)
    mask_resized = Image.fromarray(mask).resize(orig.shape[1::-1], Image.NEAREST)
    mask_np = np.array(mask_resized)

    overlay = orig.copy()  # Start with original image
    active = mask_np.sum(axis=2) > 0  # Boolean mask: where segmentation exists
    # Blend: alpha * original + (1-alpha) * mask
    overlay[active] = (alpha * orig[active] + (1 - alpha) * mask_np[active]).astype(np.uint8)
    return overlay

# Apply overlay
overlay = overlay_seg(img_pil, seg_rgb, alpha=0.6)

# Display final result
plt.figure(figsize=(8, 12))
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay: Girl (Purple) + Bicycle (Yellow)')
plt.show()

# =============================================
# 3-CLASS SEGMENTATION: Girl (Red) + Bicycle (Yellow) + BG (Black)
# =============================================

import torch
import torchvision.transforms as T
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# -------------------------------
# 1. 3-Class Decode Function (Red, Yellow, Black)
# -------------------------------
def decode_3class(image):
    colors = np.array([
        [0,   0,   0],    # 0: Background → Black
        [255, 0,   0],    # 1: Girl → Red
        [255, 255, 0]     # 2: Bicycle → Yellow
    ], dtype=np.uint8)

    h, w = image.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        rgb[image == i] = colors[i]
    return rgb

# -------------------------------
# 2. Load Image
# -------------------------------
image_path = '/content/drive/MyDrive/Girl-with-cycle.jpg'  # Update path if needed
img_pil = Image.open(image_path).convert('RGB')
img_np = np.array(img_pil)

plt.figure(figsize=(8, 12))
plt.imshow(img_np)
plt.axis('off')
plt.title('Original Image')
plt.show()

# -------------------------------
# 3. Preprocess
# -------------------------------
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img_pil).unsqueeze(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 4. Load DeepLabV3 (VOC pretrained)
# -------------------------------
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()

with torch.no_grad():
    output = model(input_tensor.to(device))['out'][0]  # [21, 224, 224]
    pred = output.argmax(0).cpu().numpy()  # (224, 224)

# -------------------------------
# 5. Base 3-class: BG=0, Girl=1
# -------------------------------
seg = np.zeros_like(pred, dtype=np.uint8)
seg[pred == 0] = 0   # background
seg[pred == 15] = 1  # person → will be red

# -------------------------------
# 6. Detect YELLOW BICYCLE using HSV
# -------------------------------
img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

# Yellow range in HSV
lower_yellow = np.array([20, 70, 70])
upper_yellow = np.array([40, 255, 255])

yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

# Resize to model size
yellow_mask_resized = cv2.resize(yellow_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
yellow_mask_resized = yellow_mask_resized.astype(bool)

# Clean small noise
kernel = np.ones((3,3), np.uint8)
yellow_mask_clean = cv2.morphologyEx(yellow_mask_resized.astype(np.uint8), cv2.MORPH_OPEN, kernel)
yellow_mask_clean = yellow_mask_clean.astype(bool)

# Assign bicycle only where not person
bicycle_mask = yellow_mask_clean & (seg != 1)
seg[bicycle_mask] = 2

# -------------------------------
# 7. Decode to RGB (Red, Yellow, Black)
# -------------------------------
seg_rgb = decode_3class(seg)

plt.figure(figsize=(8, 12))
plt.imshow(seg_rgb)
plt.axis('off')
plt.title('Segmentation: Red=Girl, Yellow=Bicycle, Black=Background')
plt.show()

# -------------------------------
# 8. Overlay on Original
# -------------------------------
def overlay_seg(orig, mask, alpha=0.6):
    orig = np.array(orig)
    mask_resized = Image.fromarray(mask).resize(orig.shape[1::-1], Image.NEAREST)
    mask_np = np.array(mask_resized)

    overlay = orig.copy()
    active = mask_np.sum(axis=2) > 0
    overlay[active] = (alpha * orig[active] + (1 - alpha) * mask_np[active]).astype(np.uint8)
    return overlay

overlay = overlay_seg(img_pil, seg_rgb, alpha=0.6)

plt.figure(figsize=(8, 12))
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay: Red=Girl, Yellow=Bicycle')
plt.show()
