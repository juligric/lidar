"""
LAIKA en vivo - Cámara del celular (app SimpleIPCamera por WiFi) en tiempo real
--------------------------------------------------------------------------------
Muestra dos paneles en una ventana: la cámara en vivo y, al lado, la
silueta reconstruida cuadro a cuadro (mismo estilo que laika_foto.py).

Cómo usarlo:
    1. Instalá "SimpleIPCamera" en el iPhone y conectalo a la MISMA WiFi que la Mac.
    2. Abrí la app, tocá "Start Server" y anotá la dirección que muestra
       (ej. http://192.168.0.4:8080/). Poné esa URL en FUENTE_CAMARA abajo.
    3. Corré:  python laika_video.py
    4. Presioná 'q' en la ventana para salir.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from laika_foto import color_por_clase

FUENTE_CAMARA = "http://192.168.0.4:8080/"   # URL del stream de SimpleIPCamera
ANCHO_PANEL = 640    # ancho de cada panel en la ventana (solo para mostrar)


def recrear_frame(resultados, modelo):
    """Arma el canvas de silueta para un frame, igual que recrear() en laika_foto.py."""
    imagen_original = resultados.orig_img
    canvas = np.full_like(imagen_original, 255)
    mascaras = resultados.masks.xy if resultados.masks is not None else None

    for i, caja in enumerate(resultados.boxes):
        nombre = modelo.names[int(caja.cls[0])]
        color = color_por_clase(nombre)
        x1, y1, x2, y2 = [int(v) for v in caja.xyxy[0].tolist()]

        mascara = np.zeros(imagen_original.shape[:2], dtype=np.uint8)
        if mascaras is not None:
            puntos = mascaras[i].astype(np.int32)
            cv2.fillPoly(mascara, [puntos], 1)
            canvas[mascara == 1] = color
            cv2.polylines(canvas, [puntos], True, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(mascara, (x1, y1), (x2, y2), 1, thickness=-1)
            canvas[mascara == 1] = color
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(canvas, nombre, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return canvas


def redimensionar(imagen, ancho):
    alto = int(imagen.shape[0] * ancho / imagen.shape[1])
    return cv2.resize(imagen, (ancho, alto))


def main():
    modelo = YOLO("yolov8m-seg.pt")   # medium: punto medio entre fluidez y precisión de contorno

    cap = cv2.VideoCapture(FUENTE_CAMARA)
    if not cap.isOpened():
        print(f"No pude abrir el stream en {FUENTE_CAMARA}. Revisá que la app SimpleIPCamera esté corriendo y la URL sea correcta.")
        return

    print("Mostrando cámara + silueta en vivo. Presioná 'q' en la ventana para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        resultados = modelo(frame, verbose=False)[0]
        canvas = recrear_frame(resultados, modelo)

        panel_izq = redimensionar(frame, ANCHO_PANEL)
        panel_der = redimensionar(canvas, ANCHO_PANEL)
        combinado = np.hstack([panel_izq, panel_der])

        cv2.imshow("LAIKA en vivo - camara | silueta", combinado)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
