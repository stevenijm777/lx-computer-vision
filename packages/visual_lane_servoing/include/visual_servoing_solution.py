from typing import Tuple

import numpy as np
import cv2
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.ground_projection.types import GroundPoint


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    height = shape[0]
    width = shape[1] // 2 
    steer_matrix_left = np.zeros(shape)
    
    # --- IZQUIERDA (Línea Amarilla) ---
    # Mantenemos tu valor de -0.3, funciona bien para rectas.
    # Gradiente: 0 (borde) -> -0.3 (centro)
    steer_unit = np.linspace(0, -0.3, width)
    
    steer_matrix_left[:, :width] = steer_unit
    
    return steer_matrix_left

def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    height = shape[0]
    width = shape[1] // 2
    steer_matrix_right = np.zeros(shape)
    
    # --- DERECHA (Línea Blanca) ---
    # AQUÍ ESTÁ EL CAMBIO CRÍTICO:
    # Antes tenías 0.1 (muy débil). Lo subimos a 0.5 o 0.6.
    # Esto hace que la línea blanca "empuje" al robot con fuerza hacia la izquierda
    # cuando entra en una intersección sin línea amarilla.
    
    fuerza_blanca = 0.6  # Prueba con 0.5, 0.6 o hasta 0.8 si sigue chocando
    
    # Gradiente: fuerza_blanca (centro) -> 0 (borde derecho)
    steer_unit = np.linspace(fuerza_blanca, 0, width)
    
    steer_matrix_right[:, width:] = steer_unit
    
    return steer_matrix_right

def detect_lane_markings(image: np.ndarray, projector: GroundProjector) -> Tuple[np.ndarray, np.ndarray]:
    
    # 1. PARAMETROS (Mantenemos los tuyos que estaban bien)
    sigma = 5
    threshold = 25
    
    # 2. COLORES (Amplíamos un poco el amarillo para asegurar detección)
    white_lower_hsv = np.array([0, 0, 100])         
    white_upper_hsv = np.array([179, 60, 255])   
    
    # Amarillo ampliado (Hue de 10 a 40 en vez de 15-35)
    yellow_lower_hsv = np.array([10, 50, 50])        
    yellow_upper_hsv = np.array([40, 255, 255]) 

    h, w, _ = image.shape
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. HORIZONTE (Tu lógica estaba bien)
    if projector is not None:
        try:
            far_away_point_ground = GroundPoint(x=10000000, y=0)
            normalized_vector = projector.ground2vector(far_away_point_ground)
            far_away_point_image = projector.camera.vector2pixel(normalized_vector)
            horizon = far_away_point_image.as_integers()[0]
        except:
            horizon = h // 2
    else:
        horizon = h // 2

    mask_ground = np.zeros((h, w), dtype=np.uint8)
    # Reducimos el buffer de +30 a +10 para no cortar líneas lejanas útiles
    mask_ground[int(horizon + 10) :, :] = 1 

    # 4. FILTROS
    img_gaussian_filter = cv2.GaussianBlur(imggray, (0, 0), sigma)
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)

    # Magnitud
    Gmag = np.sqrt(sobelx * sobelx + sobely * sobely)
    mask_mag = Gmag > threshold 

    # Colores
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    # División Izq/Der
    mask_left = np.ones(sobelx.shape)
    mask_left[:, int(w / 2) :] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:, : int(w / 2)] = 0

    # Dirección Sobel X (Esto es lo más importante para saber izq/der)
    mask_sobelx_neg = sobelx < 0
    mask_sobelx_pos = sobelx > 0
    
    # --- CAMBIO CLAVE AQUÍ ---
    # Quitamos 'mask_sobely_neg' de la multiplicación. 
    # Esto hará que detecte líneas aunque estén un poco chuecas o el robot vibre.
    
    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_white

    return mask_left_edge, mask_right_edge