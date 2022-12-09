import cv2
import numpy as np
import utils
import detectObject

webcam = True
img_path = "src/1.jpg"
video_path = "C:\myvid\prueba3.mp4"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)
scale = 2
width_background = 279.4 * scale
height_background = 215.9 * scale

if webcam:
    fps = 30
    velocidad_reproduccion = int(1000 / fps)  # milisegundos
    # Se crea el objeto que representa la fuente de video
    video = cv2.VideoCapture(video_path)

    # Si no se ha podido acceder a la fuente de video se sale del programa
    if not video.isOpened():
        print("No es posible abrir el archivo")
        exit()
    while True:
        # Se captura la imagen frame a frame
        ret, frame = video.read()

        # Código para detectar los objetos
        detectObject.classifyObjects(frame, width_background, height_background, scale)

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
    img = cv2.imread(img_path)
    # Código para detectar los objetos
    detectObject.classifyObjects(img, width_background, height_background, scale)
    cv2.waitKey(0)
