
# Add the path to torchvision - change as needed
import sys
sys.path.insert(0, '/opt/torchvision/torchvision' ) 

# Choose an image to pass through the model
#test_image = 'images/dog.png'

#---------------------------------------------------------------------
import cv2
from uuid import uuid4

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 30fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


# def gstreamer_pipeline(
#     capture_width=3280,
#     capture_height=2464,
#     display_width=820,
#     display_height=616,
#     framerate=21,
#     flip_method=0,
# ):
#     return (
#         "nvarguscamerasrc ! "
#         "video/x-raw(memory:NVMM), "
#         "width=(int)%d, height=(int)%d, "
#         "format=(string)NV12, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink"
#         % (
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )


# def take_picture():

#     cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
#     if cap.isOpened():
#         cv2.namedWindow("Test", cv2.WINDOW_AUTOSIZE)
#         while cv2.getWindowProperty("Test", 0) >= 0:
#             ret, img = cap.read()
#             cv2.imshow("Test", img)
#             keyCode = cv2.waitKey(30) & 0xFF
#             # Stop the program on the ESC key
#             if keyCode == 27:
#                 break
#             if keyCode == 32:
#                 cv2.imwrite(f"images/{uuid4()}.png",img)
#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("Unable to open camera")
    
#-------------------------------------------------------------------

# Imports
import torch, json
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
import os, time

# Import matplotlib and configure it for pretty inline plots
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# Prepare the labels
path = os.getcwd()
with open(path+"\\cnn_data\\imagenet-simple-labels.json") as f:
    labels = json.load(f)

# First prepare the transformations: resize the image to what the model was trained on and convert it to a tensor
data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# Load the image

#take_picture()

f = []
mypath = path+'\\cnn_data'
for (dirpath, dirnames, filenames) in os.walk(mypath):
    f.extend(filenames)

f = [fn for fn in f if 'png' in fn]

# Download the model if it's not there already. It will take a bit on the first run, after that it's fast
model = models.resnet50(pretrained=True)
# Send the model to the GPU
# model.cuda()
# Set layers such as dropout and batchnorm in evaluation mode
model.eval();

for image_path in f:

    start_time = time.time()
    
    image = Image.open(mypath+'/'+image_path)
    # plt.imshow(image), plt.xticks([]), plt.yticks([])

    # # Now apply the transformation, expand the batch dimension, and send the image to the GPU
    image = data_transform(image).unsqueeze(0)

    # Get the 1000-dimensional model output
    out = model(image)
    end_time = time.time()
    # Find the predicted class
    print(f"Predicted class for {image_path.split('/')[-1]} is: {labels[out.argmax()]}, done in time : {(end_time - start_time):.2f} s")


