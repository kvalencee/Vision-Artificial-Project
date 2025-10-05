"""
Transformaciones de rotación de imágenes
"""

from PIL import Image


def rotate_image(image, angle, expand=True):
    """
    Rotar imagen un ángulo específico

    Args:
        image: PIL Image
        angle: Ángulo de rotación en grados (positivo = antihorario)
        expand: Si True, expande la imagen para que quepa toda la rotación

    Returns:
        PIL Image rotada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.rotate(angle, expand=expand, resample=Image.BICUBIC)


def quick_rotate(image, angle):
    """
    Rotación rápida en múltiplos de 90 grados

    Args:
        image: PIL Image
        angle: Ángulo (debe ser -90, 90, 180, -180, 270, -270)

    Returns:
        PIL Image rotada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    # Normalizar ángulo
    angle = angle % 360
    if angle > 180:
        angle -= 360

    # Usar transpose para rotaciones de 90 grados (más eficiente)
    if angle == 90 or angle == -270:
        return image.transpose(Image.ROTATE_90)
    elif angle == -90 or angle == 270:
        return image.transpose(Image.ROTATE_270)
    elif angle == 180 or angle == -180:
        return image.transpose(Image.ROTATE_180)
    else:
        # Si no es múltiplo de 90, usar rotate normal
        return image.rotate(angle, expand=True, resample=Image.BICUBIC)


def rotate_90_clockwise(image):
    """Rotar 90 grados en sentido horario"""
    return image.transpose(Image.ROTATE_270)


def rotate_90_counterclockwise(image):
    """Rotar 90 grados en sentido antihorario"""
    return image.transpose(Image.ROTATE_90)


def rotate_180(image):
    """Rotar 180 grados"""
    return image.transpose(Image.ROTATE_180)