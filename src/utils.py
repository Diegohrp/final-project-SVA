import cv2
import numpy as np
import math


def getContours(
    img_original,
    threshold=[100, 100],
    showCanny=False,
    minArea=1000,
    maxArea=500000,
    filter=0,
    draw=False,
):
    img = img_original.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 1)
    canny = cv2.Canny(blur, threshold[0], threshold[1])
    kernel = np.ones((5, 5))
    dilate = cv2.dilate(canny, kernel, iterations=3)
    erode = cv2.erode(dilate, kernel, iterations=2)
    if showCanny:
        cv2.imshow("img_erode", erode)

    contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        # print(area)
        if area > minArea and area < maxArea:
            perimeter = cv2.arcLength(contour, True)
            # Corner points 0.02 resolution
            corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            bounding_box = cv2.boundingRect(corners)

            # Rectangles have 4 corner points, we can define that in the filter
            if filter > 0:

                if len(corners) >= filter:
                    final_contours.append(
                        [len(corners), area, corners, bounding_box, contour]
                    )
            else:
                final_contours.append(
                    [len(corners), area, corners, bounding_box, contour]
                )

    # sort by area descending order
    # x:x[1] = area in the list
    final_contours = sorted(final_contours, key=lambda x: x[1], reverse=True)

    if draw:
        for contour in final_contours:
            cv2.drawContours(img, contour[4], -1, (0, 0, 255), 3)
    return img, final_contours


def orderCorners(points):
    # print(points.shape)  # (4,1,2) 4 items of 1 row and 2 columns
    points = points.reshape((points.shape[0], 2))  # matrix 4x2
    new_points = np.zeros_like(points)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def cutImg(img_original, points, width, height):
    pad = 10
    img = img_original.copy()
    points = orderCorners(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # print(matrix)
    cut_img = cv2.warpPerspective(
        img, matrix, (int(math.ceil(width)), int(math.ceil(height)))
    )

    cut_img = cut_img[pad : cut_img.shape[0] - pad, pad : cut_img.shape[1] - pad]
    return cut_img


def calculateDistance(pts1, pts2, scale):
    x1, y1, x2, y2 = pts1[0], pts1[1], pts2[0], pts2[1]
    return math.sqrt(((x2 - x1) // scale) ** 2 + ((y2 - y1) // scale) ** 2)
