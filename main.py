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

# Opens an image; img is name of file formatted as a string
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

# Opens a video; vid is name of file formatted as a string
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

# Overlays a rectangle and text on top of an image of choice;
def overlayRect(image1, bottomRightx, bottomRighty, topLeftx, topLefty, text, rectColor = (255, 0, 0), textFontColor = (255, 255, 255)):
    image = cv2.imread(image1)
    rectImage = cv2.rectangle(image, (bottomRightx, bottomRighty), (topLeftx, topLefty), rectColor, 2)
    textImage = cv2.putText(rectImage, text,(topLeftx, topLefty), 1,
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

def detectLines(img):
    img = cv2.imread("{0}".format(img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)

    for r_theta in lines:
        arr = np.array(r_theta[0], dtype = np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('linesDetected.jpg', img)
    openImage("linesDetected.jpg")
    
def grayscale(img):
    image = cv2.imread(img)
    wn = "Gray Scale {0}".format(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow(wn, gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def callgray(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.jpg", gray)
    return cv2.imread(gray.jpg)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def subtractImages(img1, img2):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    if image1 is None:
        print(f"Error loading image1: {img1}")
        return
    if image2 is None:
        print(f"Error loading image2: {img2}")
        return

    if image1.shape != image2.shape:
        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    if image1.shape[2] == 3 and image2.shape[2] != 3:
        gray1 = callgray(image1)
        sub = cv2.subtract(gray1, image2)
        cv2.imshow("Subtracted Image", sub)
    elif image2.shape[2] == 3 and image1.shape[2] != 3:
        gray2 = callgray(image2)
        sub = cv2.subtract(image1, gray2)
        cv2.imshow("Subtracted Image", sub)
    elif image1.shape[2] == 3 and image1.shape[2] == 3:
        sub = cv2.subtract(image1, image2)
        cv2.imshow("Subtracted Image", sub)
    elif image1.shape[2] != 3 and image2.shape[2] != 3:
        sub = cv2.subtract(image1, image2)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def bitwiseOperator(img1, img2, operator, mask = None):
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
    operations = {"and": cv2.bitwise_and(image2, image1, mask), "or": cv2.bitwise_or(image2, image1, mask),
                  "not" : "nothing", "xor": cv2.bitwise_xor(image1, image2, mask)}

    if operator in operations:
        if operator != "not":
            cv2.imshow("Bitwise {0}".format(operator), operations[operator])
        else:
            x = input("img1 or img2 ")
            if x == "img1":
                cv2.imshow("Bitwise {0}".format(operator), cv2.bitwise_not(image1, mask))
            elif x == "img2":
                cv2.imshow("Bitwise {0}".format(operator), cv2.bitwise_not(image2, mask))
            else:
                return "That is not a valid image option"

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def blurring(img, blur, scale = 1):
    image = cv2.imread(img)
    if blur == "gaussian":
        Guassian = cv2.GaussianBlur(image, (7,7), 0)
        resized_image = rescaleFrame(Guassian, scale)
        cv2.imshow("Gaussian Blurring", resized_image)
        cv2.waitKey(0)
    elif blur == "median":
        median = cv2.medianBlur(image, 5)
        resized_image = rescaleFrame(median, scale)
        cv2.imshow("Median Blurring", resized_image)
        cv2.waitKey(0)
    elif blur == "bilateral":
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        resized_image = rescaleFrame(bilateral, scale)
        cv2.imshow("Bilateral Blurring", resized_image)
        cv2.waitKey(0)
    else:
        return "That is not a valid blur; please choose between gaussian, median, or bilateral."
    cv2.destroyAllWindows()


