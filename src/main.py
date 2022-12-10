import cv2
import numpy as np
import detectObject

webcam = True  # True para vídeo, False para imagen
img_path = "src/2.jpg"
video_path = "C:\myvid\prueba4.mp4"
scale = 2
# Dimensiones del fondo, hoja tamaño carta
width_background = 279.4 * scale
height_background = 215.9 * scale
# Colores de texto y recuadros
cards_color = (244, 98, 0)
coins_color = (37, 58, 32)


if webcam:  # El código se puede probar con un vídeo o una imagen
    fps = 30
    velocidad_reproduccion = int(1000 / fps)  # milisegundos
    # Se crea el objeto que representa la fuente de video
    video = cv2.VideoCapture(video_path)  # Recibe la ruta absoluta del vídeo

    # Si no se ha podido acceder a la fuente de video se sale del programa
    if not video.isOpened():
        print("No es posible abrir el archivo")
        exit()
    while True:
        # Se captura la imagen frame a frame
        ret, frame = video.read()

        # Código para detectar los objetos
        try:
            detectObject.classifyObjects(
                frame,
                width_background,
                height_background,
                scale,
                [cards_color, coins_color],
            )
        except:
            pass

        # Cuando el video termina, se sale del bucle
        if not ret:
            print("Reproduccion finalizada")
            break

        if cv2.waitKey(velocidad_reproduccion) == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()

else:
    # La detección se hace por medio de una imagen
    image = cv2.imread(img_path)
    # Código para detectar los objetos
    detectObject.classifyObjects(
        image,
        width_background,
        height_background,
        scale,
        [cards_color, coins_color],
    )
    cv2.waitKey(0)
