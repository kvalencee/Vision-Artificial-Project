"""
Transformaciones de volteo (flip) de imágenes
"""

from PIL import Image


def flip_image(image, direction='horizontal'):
    """
    Voltear imagen en dirección horizontal o vertical

    Args:
        image: PIL Image
        direction: 'horizontal' o 'vertical'

    Returns:
        PIL Image volteada

    Raises:
        TypeError: Si image no es PIL Image
        ValueError: Si direction no es válida
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    if direction not in ['horizontal', 'vertical']:
        raise ValueError("La dirección debe ser 'horizontal' o 'vertical'")

    if direction == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:  # vertical
        return image.transpose(Image.FLIP_TOP_BOTTOM)


def flip_horizontal(image):
    """
    Voltear imagen horizontalmente (espejo izquierda-derecha)

    Args:
        image: PIL Image

    Returns:
        PIL Image volteada horizontalmente
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.transpose(Image.FLIP_LEFT_RIGHT)


def flip_vertical(image):
    """
    Voltear imagen verticalmente (espejo arriba-abajo)

    Args:
        image: PIL Image

    Returns:
        PIL Image volteada verticalmente
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.transpose(Image.FLIP_TOP_BOTTOM)


def flip_both(image):
    """
    Voltear imagen tanto horizontal como verticalmente
    (equivalente a rotación de 180 grados)

    Args:
        image: PIL Image

    Returns:
        PIL Image volteada en ambas direcciones
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)


# Alias para compatibilidad
mirror_horizontal = flip_horizontal
mirror_vertical = flip_vertical