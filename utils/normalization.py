"""
Utilidades para normalización de imágenes
Basado en técnicas del libro de Visión por Computador
"""

import numpy as np


def normalize_to_uint8(image):
    """
    Normalizar imagen a rango [0, 255] usando min-max normalization

    Args:
        image: numpy array con valores arbitrarios

    Returns:
        numpy array uint8 en rango [0, 255]
    """
    if image.dtype != np.float64:
        image = image.astype(np.float64)

    min_val = np.min(image)
    max_val = np.max(image)

    # Caso especial: imagen uniforme
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    # Normalización min-max
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def normalize_gradient(magnitude):
    """
    Normalizar magnitud del gradiente según el libro

    Args:
        magnitude: numpy array con valores de magnitud

    Returns:
        numpy array normalizado
    """
    return normalize_to_uint8(magnitude)


def apply_threshold(magnitude, threshold):
    """
    Aplicar umbralización según Ecuación (6.5) del libro:
    g(x,y) = {1 si G[f(x,y)] > T, 0 si G[f(x,y)] ≤ T}

    Args:
        magnitude: Magnitud del gradiente
        threshold: Valor de umbral T

    Returns:
        Imagen binaria umbralizada
    """
    magnitude_norm = normalize_to_uint8(magnitude)
    return np.where(magnitude_norm > threshold, 255, 0).astype(np.uint8)


def angle_to_image(angle):
    """
    Convertir ángulos del gradiente a imagen visualizable
    Normaliza ángulos de [-π, π] a [0, 255]

    Args:
        angle: numpy array con ángulos en radianes

    Returns:
        numpy array uint8 visualizable
    """
    # Normalizar de [-π, π] a [0, 255]
    angle_normalized = ((angle + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    return angle_normalized


def cos_theta_to_image(cos_theta):
    """
    Convertir cos(θ) de Frei-Chen a imagen visualizable

    Args:
        cos_theta: Valores de coseno entre [0, 1]

    Returns:
        numpy array uint8 visualizable
    """
    # cos_theta ya está en [0, 1], solo escalar a [0, 255]
    return (cos_theta * 255).astype(np.uint8)


def clip_and_normalize(image, min_val=0, max_val=255):
    """
    Recortar valores fuera de rango y normalizar

    Args:
        image: numpy array
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido

    Returns:
        numpy array normalizado y recortado
    """
    clipped = np.clip(image, min_val, max_val)
    return clipped.astype(np.uint8)