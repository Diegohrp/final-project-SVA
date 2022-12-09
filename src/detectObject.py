import cv2
import utils


def textDecoration(img, labels, x, xdiff, y, color):
    ydiff = 30
    for attrib, value in labels:
        cv2.putText(
            img,
            f"{attrib}: {value}",
            (x + xdiff, y - ydiff),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.8,
            color,
            1,
        )
        ydiff -= 14


def detectObj(objs_img, objs_contours, scale, color):
    if len(objs_contours) != 0:
        for obj in objs_contours:
            # Draw border
            x, y, w, h = obj[3]
            # Es una tarjeta si tiene 4 esquinas
            if len(obj[2]) == 4:
                cv2.rectangle(objs_img, (x, y), (x + w, y + h), color, 2)
                obj_points = utils.orderCorners(obj[2])
                obj_width = round(
                    utils.calculateDistance(obj_points[0], obj_points[1], scale) / 10, 1
                )
                obj_height = round(
                    utils.calculateDistance(obj_points[0], obj_points[2], scale) / 10, 1
                )

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

            if len(obj[2]) >= 8:
                cv2.rectangle(objs_img, (x, y), (x + w, y + h), color, 2)
                obj_points = obj[2].reshape(obj[2].shape[0], 2)
                diameter = round(
                    utils.calculateDistance(obj_points[0], obj_points[5], scale) / 10, 1
                )
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
    return len(objs_contours)


def classifyObjects(img, width_background, height_background, scale):

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("RESIZE", img)

    img_contour, final_contours = utils.getContours(
        img, [55, 55], showCanny=True, draw=False, minArea=30000, filter=4
    )

    # print(len(final_contours))

    if len(final_contours) != 0:
        background = final_contours[0][2]  # Esquinas
        cut_img = utils.cutImg(img, background, width_background, height_background)
        # cv2.imshow("letter size", cut_img)

        # Detectar tarjetas
        cards_img, cards_contours = utils.getContours(
            cut_img,
            [55, 55],
            showCanny=False,
            draw=False,
            minArea=10000,
            filter=4,
        )
        cards_quantity = detectObj(cut_img, cards_contours, scale, (0, 0, 255))

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
        coins_quantity = detectObj(cut_img, coins_contours, scale, (255, 0, 0))

        xb, yb, wb, hb = final_contours[0][3]
        print([xb, yb, wb, hb])
        if cards_quantity:
            cv2.putText(
                cut_img,
                f"Tarjetas: {cards_quantity}",
                (xb + wb // 2, yb + hb - 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.8,
                (0, 0, 255),
                1,
            )
        if coins_quantity:
            cv2.putText(
                cut_img,
                f"Monedas: {coins_quantity}",
                (xb + wb // 2, yb + hb - 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.8,
                (255, 0, 0),
                1,
            )

        cv2.imshow("cards", cut_img)
