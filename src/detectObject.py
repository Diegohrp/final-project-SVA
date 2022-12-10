import cv2
import utils

# Función para generar los textos de los recuadros cuando se
# identifica un objeto (tipo de obj, dimensiones, etc.)
def textDecoration(img, labels, x, xdiff, y, color):
    ydiff = 30  # Espaciado inicial en y
    for attrib, value in labels:
        cv2.putText(
            img,
            f"{attrib}: {value}",
            (x + xdiff, y - ydiff),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.85,
            color,
            1,
        )
        ydiff -= 14  # Cada objeto tiene un espaciado en y de 14px entre sí


def detectObj(objs_img, objs_contours, scale, color):
    if len(objs_contours) != 0:
        for obj in objs_contours:
            # box_bounding, para dibujar los recuadros
            x, y, w, h = obj[3]
            # Es una tarjeta si tiene 4 esquinas
            if len(obj[2]) == 4:
                cv2.rectangle(objs_img, (x, y), (x + w, y + h), color, 2)
                obj_points = utils.orderCorners(obj[2])
                # Obtiene las dimensiones de la targeta a partir de 2 puntos, 2 esquinas.
                obj_width = round(
                    utils.calculateDistance(obj_points[0], obj_points[1], scale) / 10, 1
                )
                obj_height = round(
                    utils.calculateDistance(obj_points[0], obj_points[2], scale) / 10, 1
                )
                # Se llama la función para generar los textos
                textDecoration(
                    objs_img,
                    [
                        ["Obj", "Tarjeta."],
                        ["Ancho", f"{obj_width}cm."],
                        ["Alto", f"{obj_height}cm."],
                    ],
                    x,
                    20,  # diff
                    y,
                    color,
                )
            # Si tiene 8 esquinas o más es una moneda
            if len(obj[2]) >= 8:
                cv2.rectangle(objs_img, (x, y), (x + w, y + h), color, 2)
                obj_points = obj[2].reshape(obj[2].shape[0], 2)  # matriz 4x2
                # Calculamos el diámetro a partir de 2 puntos opuestos
                diameter = round(
                    utils.calculateDistance(obj_points[0], obj_points[5], scale) / 10, 1
                )
                # Se llama la función para generar los textos
                textDecoration(
                    objs_img,
                    [
                        ["Obj", "Moneda."],
                        ["D", f"{diameter}cm."],
                    ],
                    x,
                    -20,  # xdiff
                    y,
                    color,
                )
    # Se retorna el tamaño del array objs_contours, el cual contiene los contornos de cada
    # objeto identificado, por lo tanto aquí se cuenta cuantos objetos de un tipo hay.
    return len(objs_contours)


# Funcín principal del programa, es la que se llama en el main
def classifyObjects(img, width_background, height_background, scale, colors):
    # Redimensiona la imagen a la mitad
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Original", img)

    img_contour, final_contours = utils.getContours(
        img, [55, 55], showCanny=False, draw=False, minArea=30000, filter=4
    )
    # Si identificó los bordes del fondo, ejecuta
    if len(final_contours) != 0:
        background = final_contours[0][2]  # Esquinas (4)

        # Recortar la img o frame a las dimensiones de una hoja carta
        cut_img = utils.cutImg(img, background, width_background, height_background)

        # Detectar tarjetas
        cards_img, cards_contours = utils.getContours(
            cut_img,
            [55, 55],
            showCanny=False,
            draw=False,
            minArea=10000,
            filter=4,
        )
        # Contar tarjetas, dibujar cuadro y texto
        cards_quantity = detectObj(cut_img, cards_contours, scale, colors[0])

        # Detectar monedas
        coins_img, coins_contours = utils.getContours(
            cards_img,
            [55, 55],
            showCanny=False,
            draw=False,
            minArea=1000,
            maxArea=3500,
            filter=8,
        )
        # Contar monedas, dibujar cuadro y texto
        coins_quantity = detectObj(cut_img, coins_contours, scale, colors[1])

        xb, yb, wb, hb = final_contours[0][3]

        # Despliega los textos en la imagen con la cantidad de monedas y tarjetas
        if cards_quantity:
            textDecoration(
                cut_img,
                [["Tarjetas", cards_quantity]],
                xb,
                wb // 2,
                400,
                colors[0],
            )

        if coins_quantity:
            textDecoration(
                cut_img,
                [["Monedas", coins_quantity]],
                xb,
                wb // 2,
                420,
                colors[1],
            )
        cv2.imshow("Objetos", cut_img)
