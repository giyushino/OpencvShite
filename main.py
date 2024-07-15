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

def openImage(img):
    image = cv2.imread("{0}".format(img))
    window_name = "{0}".format(img)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def rescaleFrame(frame, scale= 1):
    # rescales images, video, live-video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def openVideo(vid):
    capture = cv2.VideoCapture("{}".format(vid))
    while True:
        isTrue, frame = capture.read()
        resized_frame = rescaleFrame(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF==ord('d'):
            break
    capture.release()
    cv2.destroyAllWindows()

def overlayRect(image1, upperx, uppery, lowerx, lowery, text, rectColor = (255, 0, 0), textFontColor = (255, 255, 255)):
    image = cv2.imread(image1)
    rectImage = cv2.rectangle(image, (upperx, uppery), (lowerx, lowery), rectColor, 2)
    textImage = cv2.putText(rectImage, text,(lowerx, lowery), 1,
                            1, textFontColor, 2)
    cv2.imshow(image1, textImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Overlay 2 images; w1 is weight of first image, w2 is weight of second photo
def overlayImages(img1, img2, w1, w2, gamma = 0):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    if image1 is None:
        print(f"Error loading image1: {img1}")
        return
    if image2 is None:
        print(f"Error loading image2: {img2}")
        return

    if image1.shape != image2.shape:
        print("Images have different dimensions. Resizing image2 to match image1.")
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    weightedSum = cv2.addWeighted(image1, w1, image2, w2, gamma)
    cv2.imshow("Weighted Image", weightedSum)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


openImage("coca-cola-logo.png")

overlayImages("coca-cola-logo.png", "monkey.png", 0.8, 0.2)

overlayRect("coca-cola-logo.png", 250, 250, 20, 20, "Hello World!")

openVideo("lp_image.mov")

