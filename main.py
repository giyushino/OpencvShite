import cv2
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve
from IPython.display import Image

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

def rescaleFrame(frame, scale=0.75): 
    # rescales images, video, live-video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Download if asset ZIP does not exist.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

# Open Image
path = r'C:\Users\allan\PycharmProjects\opencv\ImageName'
image = cv2.imread(path)
window_name = "ImageName"
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Overlaying two images
image1 = cv2.imread("image1")
image2 = cv2.imread("image2")

if image1.shape != image2.shape:
    print("Images have different dimensions. Resizing image2 to match image1.")
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

weightedSum = cv2.addWeighted(image1, 0.8, image2, 0.1, 0)
cv2.imshow("Weighted Image", weightedSum)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# Open Videos
capture = cv2.VideoCapture("lp_image.mov")
while True:
  isTrue, frame = capture.read()
  # resized_frame = rescaleFrame(frame)
  cv2.imshow('Video', frame)
  if cv2.waitKey(20) & 0xFF==ord('d'):
    break
capture.release()
cv2.destroyAllWindows()
