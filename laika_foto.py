"""
LAIKA - De una foto (carpeta 'fotos/') a coordenadas + réplica
--------------------------------------------------------------
1) Agarra la(s) foto(s) que pongas en la carpeta 'fotos/'.
2) Detecta los objetos con YOLOv8-seg (modelo ya entrenado en 80 clases comunes).
3) Guarda un .npy con las coordenadas de cada objeto.
4) Dibuja una réplica: la silueta real de cada objeto (según la máscara de
   segmentación que da el modelo), en un color sólido por clase, sobre fondo blanco.

Cómo usarlo:
    1. Creá una carpeta llamada 'fotos' al lado de este archivo.
    2. Poné adentro tu foto (.jpg o .png).
    3. Corré:  python laika_foto.py
    Los resultados quedan en la carpeta 'resultados/'.

Instalar (una sola vez):
    pip install ultralytics numpy opencv-python-headless
"""

import sys
import zlib
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

CARPETA_FOTOS = "fotos"
CARPETA_SALIDA = "resultados"
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def buscar_fotos(carpeta):
    """Devuelve la lista de imágenes que hay en la carpeta."""
    ruta = Path(carpeta)
    if not ruta.exists():
        ruta.mkdir(parents=True, exist_ok=True)
        print(f"Creé la carpeta '{carpeta}'. Poné una foto adentro y volvé a correr.")
        sys.exit(0)
    fotos = [p for p in sorted(ruta.iterdir()) if p.suffix.lower() in EXTENSIONES]
    if not fotos:
        print(f"No encontré ninguna foto en '{carpeta}'. Poné un .jpg o .png adentro.")
        sys.exit(0)
    return fotos


def detectar_objetos(modelo, ruta_foto):
    """Detecta objetos y devuelve un array numpy con sus coordenadas + el contorno de cada uno + el tamaño de la foto."""
    resultados = modelo(str(ruta_foto))[0]
    dtype = [("clase", "U20"), ("confianza", "f4"),
             ("x1", "f4"), ("y1", "f4"), ("x2", "f4"), ("y2", "f4"),
             ("cx", "f4"), ("cy", "f4")]
    filas = []
    contornos = []
    mascaras = resultados.masks.xy if resultados.masks is not None else None
    for i, caja in enumerate(resultados.boxes):
        nombre = modelo.names[int(caja.cls[0])]
        conf = float(caja.conf[0])
        x1, y1, x2, y2 = caja.xyxy[0].tolist()
        filas.append((nombre, conf, x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2))
        contornos.append(mascaras[i] if mascaras is not None else None)
    objetos = np.array(filas, dtype=dtype)
    return objetos, contornos, resultados.orig_img   # orig_img: foto original en BGR


PALETA_BGR = [
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
    (189, 103, 148), (75, 86, 140), (194, 119, 227), (127, 127, 127),
    (34, 189, 188), (207, 190, 23),
]


def color_por_clase(nombre):
    """Color fijo por nombre de clase (no depende de qué otras clases haya en el cuadro)."""
    indice = zlib.crc32(nombre.encode()) % len(PALETA_BGR)
    return PALETA_BGR[indice]


def recrear(objetos, contornos, imagen_original, ruta_salida):
    """Dibuja la réplica: la silueta real de cada objeto (por su máscara de
    segmentación, o su caja si no hay máscara) en un color sólido por clase."""
    canvas = np.full_like(imagen_original, 255)

    for o, contorno in zip(objetos, contornos):
        color = color_por_clase(str(o["clase"]))
        mascara = np.zeros(imagen_original.shape[:2], dtype=np.uint8)
        if contorno is not None and len(contorno) >= 3:
            puntos = contorno.astype(np.int32)
            cv2.fillPoly(mascara, [puntos], 1)
            canvas[mascara == 1] = color
            cv2.polylines(canvas, [puntos], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = int(o["x1"]), int(o["y1"]), int(o["x2"]), int(o["y2"])
            cv2.rectangle(mascara, (x1, y1), (x2, y2), 1, thickness=-1)
            canvas[mascara == 1] = color
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)

        cx, cy = int(o["cx"]), int(o["cy"])
        cv2.putText(canvas, str(o["clase"]), (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(ruta_salida), canvas)


def main():
    Path(CARPETA_SALIDA).mkdir(exist_ok=True)
    modelo = YOLO("yolov8x-seg.pt")   # se descarga solo la primera vez (~140 MB)

    for foto in buscar_fotos(CARPETA_FOTOS):
        print(f"\nProcesando {foto.name} ...")
        objetos, contornos, imagen_original = detectar_objetos(modelo, foto)

        npy = Path(CARPETA_SALIDA) / f"{foto.stem}_objetos.npy"
        png = Path(CARPETA_SALIDA) / f"{foto.stem}_replica.png"
        np.save(npy, objetos)
        recrear(objetos, contornos, imagen_original, png)

        print(f"  Detecté {len(objetos)} objetos:")
        for o in objetos:
            print(f"    - {o['clase']:12s} conf={o['confianza']:.2f}  centro=({o['cx']:.0f},{o['cy']:.0f})")
        print(f"  Coordenadas -> {npy}")
        print(f"  Réplica     -> {png}")


if __name__ == "__main__":
    main()