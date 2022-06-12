# PragnaMallikarjunaSwamy- Homework2

# import opencv, math and numpy libraries
import cv2
import numpy as np
import math


# Define a method to apply the Gaussian filter to remove noise
# as noise can be detected as edge due to sudden intensity changes
def gaussfilter(img_gray, size, k, sigma):
    gausskernel = np.zeros((size, size), np.float32)
    # Create a gaussian filter
    for i in range(size):
        for j in range(size):
            norm = math.pow(i - k, 2) + pow(j - k, 2)
            gausskernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2))) / 2 * math.pi * pow(sigma, 2)

    # Apply the Gaussian filter over the image
    sum = np.sum(gausskernel)
    kernel = gausskernel / sum
    height, width = img_gray.shape
    kernel_h, kernel_w = kernel.shape
    for i in range(int(kernel_h / 2), height - int(kernel_h / 2)):
        for j in range(int(kernel_h / 2), width - int(kernel_h / 2)):
            sum = 0
            for k in range(0, kernel_h):
                for l in range(0, kernel_h):
                    sum += img_gray[i - int(kernel_h / 2) + k, j - int(kernel_h / 2) + l] * kernel[k, l]
            img_gray[i, j] = sum
    return img_gray


# defining the canny detector function
def CannyEdgeDetector(img):

    # converting color image to grayscale using opencv
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    k = 1
    size = 2 * k + 1
    # Apply Gaussian filter with sigma = 1.5
    img_gauss = gaussfilter(img_gray, size, k, 1.5)

# Calculating the gradients using Sobel filter
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(img_gauss)
    magnitude = np.zeros(shape=(rows, columns))
    angle = np.zeros(shape=(rows, columns))

    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, img_gauss[i:i + 3, j:j + 3]))  # x direction derivative
            gy = np.sum(np.multiply(Gy, img_gauss[i:i + 3, j:j + 3]))  # y direction derivative
            # Calculate the magnitude and orientation of gradients
            magnitude[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
            angle[i+1, j+1] = np.arctan(gy/gx)

    # Get the size of the image
    height, width = img_gauss.shape
    adjpixel1_x, adjpixel1_y, adjpixel2_x, adjpixel2_y = 0, 0, 0, 0
    for x in range(width):
        for y in range(height):

            grad_ang = angle[y, x]
            # check if orientation is negative, i.e. less than 0
            # if it is, add 180 to ensure values between 0 and 180
            if grad_ang < 0:
                grad_ang = abs(grad_ang + 180)
            else:
                grad_ang = abs(grad_ang)

            # For each pixel, choose two neighbours in the same direction of gradient
            # x-axis direction
            if grad_ang <= 22.5 or grad_ang > 157.5:
                adjpixel1_x, adjpixel1_y = x - 1, y
                adjpixel2_x, adjpixel2_y = x + 1, y

            # Diagonal 45 direction
            elif 22.5 < grad_ang <= 67.5:
                adjpixel1_x, adjpixel1_y = x - 1, y - 1
                adjpixel2_x, adjpixel2_y = x + 1, y + 1

            # y-axis direction
            elif 67.5 < grad_ang <= 112.5:
                adjpixel1_x, adjpixel1_y = x - 1, y + 1
                adjpixel2_x, adjpixel2_y = x + 1, y - 1

            # Diagonal 135 direction
            elif 112.5 < grad_ang <= 157.5:
                adjpixel1_x, adjpixel1_y = x, y + 1
                adjpixel2_x, adjpixel2_y = x, y - 1

            # Non-maximum suppression -
            # If magnitude of current pixel is smaller than neighboring pixels, set it to 0
            # Otherwise keep gradient magnitude
            if width > adjpixel1_x >= 0 and height > adjpixel1_y >= 0:
                if magnitude[y, x] < magnitude[adjpixel1_y, adjpixel1_x]:
                    magnitude[y, x] = 0

            if width > adjpixel2_x >= 0 and height > adjpixel2_y >= 0:
                if magnitude[y, x] < magnitude[adjpixel2_y, adjpixel2_x]:
                    magnitude[y, x] = 0

    # Hysteresis Thresholding
    # create a numpy array of zeros to track edges
    edges = np.zeros_like(img_gauss)
    # Set a low and high threshold value
    low_th = np.int32(75)
    high_th = np.int32(100)

    for x in range(width - 1):
        for y in range(height - 1):

            grad_mag = magnitude[y, x]
            # if gradient magnitude is lower than lower threshold, ignore it
            if grad_mag < low_th:
                magnitude[y, x] = 0
            # if gradient magnitude is between lower and higher threshold, set the values in edges array to 1
            elif high_th > grad_mag >= low_th:
                edges[y, x] = 1
            # if gradient magnitude is greater than higher threshold,
            # set values in edges array to 2 and set gradient magnitude to max
            else:
                edges[y, x] == 2
                magnitude[y, x] = np.max(magnitude)

            # for every pixel with magnitude between lower and higher threshold,
            # check if any neighboring pixel has a threshold greater than higher threshold
            # if yes, change gradient magnitude of that pixel to max
            if edges[y, x] == 1:
                if (edges[y + 1, x] == 2) or (edges[y - 1, x] == 2) or (edges[y, x + 1] == 2) or (edges[y, x-1] == 2) or (edges[y + 1, x+ 1] == 2) or (edges[y - 1, x - 1] == 2) or (edges[y + 1, x -1] == 2) or (edges[y - 1, x + 1] == 2):
                    magnitude[y, x] = np.max(magnitude)

    # return the magnitude which are edges
    return magnitude


# load the image
frame = cv2.imread('tulips.png')
# call the canny method to detect edges
canny_img = CannyEdgeDetector(frame)
# display the edges in the image
cv2.imshow("Canny-Image", canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
