# Canny Edge Detector
# Problem Statement:
Detect and display edges of image using the canny edge detection algorithm.
 
# Description:
1. Edges- Edges are the parts of an image that represent the boundary or shape of an object. They are marked by sudden changes in intensity compared to the neighboring pixels.
2. Edge Detection- Edge detection involves methods that aim at identifying edges where the image intensity/brightness changes significantly or has discontinuities.
3. Canny Edge Detection Algorithm – Canny Edge Detection is a popular edge detection algorithm developed by John F. Canny in 1986. It involves 4 steps- Noise reduction, Calculating magnitude and orientation of the gradient, non-maximum suppression, and Hysteresis Thresholding.

# Canny Edge Detection:
1. Noise Reduction- This is done by first converting the color image to grayscale. Since edge detection is susceptible to noise, this is an important step. A Gaussian filter is applied to remove the noise and smooth the image.
2. Calculating magnitude and orientation of gradient- To calculate the magnitude and orientation, a Sobel filter is used to get the first derivatives in the horizontal (Gx) and vertical direction (Gy). After that, the magnitude and angle is calculated.
3. Non- maximum suppression- In this process, we need to identify the neighboring pixels in every direction. We do this as- If any orientation is negative, add 180 to make it positive and identify neighboring pixels using the table below. If the magnitude of the current pixel is smaller than the neighboring pixels for a particular direction, set it to zero, otherwise keep the magnitude.
4. Hysteresis Thresholding- In this step, we choose the strong edges and remove the weak edges. This is done by setting a low and high threshold, and any gradient magnitude that is above the high threshold will be an edge and any lower that that will be ignored. Any gradient magnitude that is between the high and low threshold will be chosen if they are connected to a pixel with magnitude greater than the high threshold.

# Working of the Program:
1. Read the image files, e.g.: Lenna.png.
2. Convert the colored image to grayscale.
3. Apply Gaussian filter with kernel 3*3 with varying sigma values to remove noise.
4. Calculate the gradients using the Sobel filter, one in the horizontal direction and one in the
vertical direction.
5. From the gradients generated, calculate the magnitude of gradient and orientation.
6. Perform non-maximum suppression by finding out neighboring pixels (Described above in
Step 3).
7. Perform hysteresis thresholding to keep only strong edges.
8. Display the canny image which shows only edges.
(More explanation is included in the comments in code- All 3 images placed in the project folder)
