import cv2
import numpy as np
import math

# Obtiene los contornos de las figuras identificadas en el fondo
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
    # Se aplica un filtro blur, cany, de dilatació y eroción para una mejor
    # detección de los bordes.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 1)
    canny = cv2.Canny(blur, threshold[0], threshold[1])
    kernel = np.ones((5, 5))
    dilate = cv2.dilate(canny, kernel, iterations=3)
    erode = cv2.erode(dilate, kernel, iterations=2)
    if showCanny:
        cv2.imshow("img_erode", erode)

    # Todos los contornos identificados en la imagen
    contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []  # array de arrays

    # Se cuenta con un array de contornos
    # Sólo los que cumplan con ciertas condiciones serán almacenados finalmente
    for contour in contours:
        area = cv2.contourArea(contour)
        # Área en un rango definido por el usuario
        if area > minArea and area < maxArea:
            perimeter = cv2.arcLength(contour, True)
            # Corner points 0.02 resolution
            corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            # Coordenadas x,y y tamaños h y w para posteriormente generar
            # Un recuadro al rededor del objeto
            bounding_box = cv2.boundingRect(corners)

            # Filtramos por el número de esquinas del objeto, este valor
            # se recibe como parámetro
            # filter = 4 para rectángulos (las tarjetas)
            # filter >=8 para los círculos (las monedas)
            if filter > 0:

                if len(corners) >= filter:
                    final_contours.append(
                        [len(corners), area, corners, bounding_box, contour]
                    )
            else:
                final_contours.append(
                    [len(corners), area, corners, bounding_box, contour]
                )

    # Ordenamos los elementos en el array a partir del área
    # El orden es descendente (mayor a menor)
    # x:x[1] = área en el array dentro del array principal
    final_contours = sorted(final_contours, key=lambda x: x[1], reverse=True)

    # En caso de ser true, dibuja los contornos de los objetos
    if draw:
        for contour in final_contours:
            cv2.drawContours(img, contour[4], -1, (0, 0, 255), 3)
    return img, final_contours


"""
    Cuando se obtienen los contornos, las esquinas no vienen en el orden adecuado
    para calcular las distancias, en esta función se ordenan las esquinas así: 

    1    2
    3    4
    Esta función aplica solo para las tarjetas
"""


def orderCorners(points):
    # print(points.shape)  # (4,1,2) 4 elementos de 1 fila y 2 columnas
    points = points.reshape((points.shape[0], 2))  # matriz 4x2
    new_points = np.zeros_like(points)
    add = points.sum(1)  # Suma ambas coordenadas de cada esquina
    # Realiza una comparación entre x's y's por medio de
    # mínimos y máximos
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


# Recorta la imagen o frame original al tamaño del fondo (la hoja carta)
def cutImg(img_original, points, width, height):
    pad = 10  # Padding para el excedente al fondo
    img = img_original.copy()
    points = orderCorners(points)  # Ordenamos las esquinas
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Se recorta la imagen a las dimensiones de una hoja carta, sólo que aquí en px
    cut_img = cv2.warpPerspective(
        img, matrix, (int(math.ceil(width)), int(math.ceil(height)))
    )
    # Se remueve el execente para una mejor estética, esto no afecta las dimensiones
    # del fondo (la hoja carta) que ya se tienen.
    cut_img = cut_img[pad : cut_img.shape[0] - pad, pad : cut_img.shape[1] - pad]
    return cut_img


# Función que calcula la distancia entre dos puntos
def calculateDistance(pts1, pts2, scale):
    x1, y1, x2, y2 = pts1[0], pts1[1], pts2[0], pts2[1]
    return math.sqrt(((x2 - x1) // scale) ** 2 + ((y2 - y1) // scale) ** 2)
