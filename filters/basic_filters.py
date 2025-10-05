"""
Filtros básicos de imagen
"""

import numpy as np
from PIL import Image


def apply_grayscale(image):
    """
    Convertir imagen a escala de grises

    Args:
        image: PIL Image

    Returns:
        PIL Image en escala de grises
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.convert("L").convert("RGB")


def apply_sepia(image):
    """
    Aplicar efecto sepia vintage

    Args:
        image: PIL Image

    Returns:
        PIL Image con efecto sepia
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    img_array = np.array(image)

    # Matriz de transformación sepia
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    # Aplicar transformación
    sepia_img = np.dot(img_array[..., :3], sepia_filter.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

    return Image.fromarray(sepia_img)


def apply_invert(image):
    """
    Invertir colores de la imagen (negativo)

    Args:
        image: PIL Image

    Returns:
        PIL Image invertida
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    img_array = np.array(image)
    inverted = 255 - img_array

    return Image.fromarray(inverted.astype(np.uint8))


def apply_warm_filter(image, intensity=0.3):
    """
    Aplicar filtro cálido (aumentar tonos naranjas/rojos)

    Args:
        image: PIL Image
        intensity: Intensidad del filtro (0.0 a 1.0)

    Returns:
        PIL Image con filtro cálido
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    img_array = np.array(image).astype(np.float32)

    # Aumentar canal rojo y reducir azul
    img_array[..., 0] = np.clip(img_array[..., 0] * (1 + intensity * 0.3), 0, 255)
    img_array[..., 2] = np.clip(img_array[..., 2] * (1 - intensity * 0.2), 0, 255)

    return Image.fromarray(img_array.astype(np.uint8))


def apply_cool_filter(image, intensity=0.3):
    """
    Aplicar filtro frío (aumentar tonos azules)

    Args:
        image: PIL Image
        intensity: Intensidad del filtro (0.0 a 1.0)

    Returns:
        PIL Image con filtro frío
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    img_array = np.array(image).astype(np.float32)

    # Aumentar canal azul y reducir rojo
    img_array[..., 2] = np.clip(img_array[..., 2] * (1 + intensity * 0.3), 0, 255)
    img_array[..., 0] = np.clip(img_array[..., 0] * (1 - intensity * 0.2), 0, 255)

    return Image.fromarray(img_array.astype(np.uint8))


def apply_posterize(image, levels=4):
    """
    Reducir número de colores (posterización)

    Args:
        image: PIL Image
        levels: Número de niveles por canal

    Returns:
        PIL Image posterizada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    if levels < 2 or levels > 8:
        levels = 4

    img_array = np.array(image)

    # Reducir niveles
    factor = 256 // levels
    posterized = (img_array // factor) * factor

    return Image.fromarray(posterized.astype(np.uint8))


def apply_solarize(image, threshold=128):
    """
    Aplicar efecto solarización

    Args:
        image: PIL Image
        threshold: Umbral de solarización

    Returns:
        PIL Image solarizada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    img_array = np.array(image)

    # Invertir píxeles por encima del umbral
    mask = img_array > threshold
    solarized = img_array.copy()
    solarized[mask] = 255 - img_array[mask]

    return Image.fromarray(solarized.astype(np.uint8))


def apply_basic_filter(image, filter_type):
    """
    Aplicar filtro básico según tipo

    Args:
        image: PIL Image
        filter_type: Tipo de filtro ('grayscale', 'sepia', 'invert', etc.)

    Returns:
        PIL Image filtrada
    """
    if filter_type == "grayscale":
        return apply_grayscale(image)
    elif filter_type == "sepia":
        return apply_sepia(image)
    elif filter_type == "invert":
        return apply_invert(image)
    elif filter_type == "warm":
        return apply_warm_filter(image)
    elif filter_type == "cool":
        return apply_cool_filter(image)
    elif filter_type == "posterize":
        return apply_posterize(image)
    elif filter_type == "solarize":
        return apply_solarize(image)
    elif filter_type == "original":
        return image.copy()
    else:
        raise ValueError(f"Filtro desconocido: {filter_type}")