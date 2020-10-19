# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread("zaman.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# canny = cv2.Canny(img, 100, 50)
#
# titles = ['image', 'roberts']
# images = [img, canny]
# for i in range(2):
#     plt.subplot(1, 2, i+1), plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#
# plt.show()
#


import cv2
import numpy as np

img = cv2.imread('sudoku.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('roberts', edges)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))

    y1 = int(y0 + 1000 * (a))

    x2 = int(x0 - 1000 * (-b))

    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('image', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

#
#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('zaman.PNG', cv2.IMREAD_GRAYSCALE)
# _, mask = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
#
# kernal = np.ones((6, 6), np.uint8)
#
# dilation = cv2.dilate(mask, kernal, iterations=2)
# erosion = cv2.erode(mask, kernal, iterations=1)
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
# closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
# mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)
# th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal)
#
# titles = ['image', 'first', 'second', 'roberts']
# images = [img, dilation, closing, mg]
#
# for i in range(4):
#     plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
#
# plt.show()


#
# # import required library
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # create a video object
# # for capture the frames.
# # for Webcamera we pass 0
# # as an argument
# cap = cv2.VideoCapture(0)
#
#
# # define a empty function
# def nothing(x):
#     pass
#
#
# # set windown name
# cv2.namedWindow('Tracking')
#
# # Creates a trackbar and attaches
# # it to the specified window
# # with nothing function
# cv2.createTrackbar("LH", "Tracking",
#                    0, 255, nothing)
# cv2.createTrackbar("LS", "Tracking",
#                    0, 255, nothing)
# cv2.createTrackbar("LV", "Tracking",
#                    0, 255, nothing)
# cv2.createTrackbar("HH", "Tracking",
#                    0, 255, nothing)
# cv2.createTrackbar("HS", "Tracking",
#                    0, 255, nothing)
# cv2.createTrackbar("HV", "Tracking",
#                    0, 255, nothing)
#
# # This drives the program
# # into an infinite loop.
# while True:
#
#     # Captures the live stream frame-by-frame
#     _, frame = cap.read()
#
#     # Converts images from BGR to HSV
#     hsv = cv2.cvtColor(frame,
#                        cv2.COLOR_BGR2HSV)
#
#     # find LH trackbar position
#     l_h = cv2.getTrackbarPos("LH",
#                              "Tracking")
#     # find LS trackbar position
#     l_s = cv2.getTrackbarPos("LS",
#                              "Tracking")
#     # find LV trackbar position
#     l_v = cv2.getTrackbarPos("LV",
#                              "Tracking")
#     # find HH trackbar position
#     h_h = cv2.getTrackbarPos("HH",
#                              "Tracking")
#     # find HS trackbar position
#     h_s = cv2.getTrackbarPos("HS",
#                              "Tracking")
#     # find HV trackbar position
#     h_v = cv2.getTrackbarPos("HV",
#                              "Tracking")
#     # create a given numpy array
#     l_b = np.array([l_h, l_s,
#                     l_v])
#     # create a given numpy array
#     u_b = np.array([h_h, h_s,
#                     h_v])
#     # create a mask
#     mask = cv2.inRange(hsv, l_b,
#                        u_b)
#     # applying bitwise_and operation
#     res = cv2.bitwise_and(frame,
#                           frame, mask=mask)
#
#     # display frame, mask
#     # and res window
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)
#
#     # wait for 1 sec
#     k = cv2.waitKey(1)
#
#     # break out of while loop
#     # if k value is 27
#     if k == 27:
#         break
#
# # release the captured frames
# cap.release()
#
# # Destroys all windows.
# cv2.destroyAllWindows()
#
#
#
#


#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread("zaman.jpg", cv2.IMREAD_GRAYSCALE)
# lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
# lap = np.uint8(np.absolute(lap))
# sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
# sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
# edges = cv2.Canny(img,100,200)
#
# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))
#
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)
#
# titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'Canny']
# images = [img, lap, sobelX, sobelY, sobelCombined, edges]
# for i in range(6):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#
# plt.show()


#
# import cv2
# import numpy as np
#
#
# def nothing(x):
#     pass
#
#
# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
#
# while True:
#     frame = cv2.imread('zaman.png')
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     l_h = cv2.getTrackbarPos("LH", "Tracking")
#     l_s = cv2.getTrackbarPos("LS", "Tracking")
#     l_v = cv2.getTrackbarPos("LV", "Tracking")
#
#     u_h = cv2.getTrackbarPos("UH", "Tracking")
#     u_s = cv2.getTrackbarPos("US", "Tracking")
#     u_v = cv2.getTrackbarPos("UV", "Tracking")
#
#     l_b = np.array([l_h, l_s, l_v])
#     u_b = np.array([u_h, u_s, u_v])
#
#     mask = cv2.inRange(hsv, l_b, u_b)
#
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     cv2.imshow("frame", frame)
#     cv2.imshow("mask", mask)
#     cv2.imshow("res", res)
#
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
#
#
# def nothing(x):
#     pass
#
#
# cap = cv2.VideoCapture(0);
#
# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
#
# while True:
#     # frame = cv2.imread('smarties.png')
#     _, frame = cap.read()
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     l_h = cv2.getTrackbarPos("LH", "Tracking")
#     l_s = cv2.getTrackbarPos("LS", "Tracking")
#     l_v = cv2.getTrackbarPos("LV", "Tracking")
#
#     u_h = cv2.getTrackbarPos("UH", "Tracking")
#     u_s = cv2.getTrackbarPos("US", "Tracking")
#     u_v = cv2.getTrackbarPos("UV", "Tracking")
#
#     l_b = np.array([l_h, l_s, l_v])
#     u_b = np.array([u_h, u_s, u_v])
#
#     mask = cv2.inRange(hsv, l_b, u_b)
#
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     cv2.imshow("frame", frame)
#     cv2.imshow("mask", mask)
#     cv2.imshow("res", res)
#
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
