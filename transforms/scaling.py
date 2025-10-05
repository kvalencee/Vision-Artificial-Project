"""
Transformaciones de escalado de imágenes
"""

from PIL import Image


def scale_image(image, scale_x=1.0, scale_y=1.0, resample=Image.LANCZOS):
    """
    Escalar imagen con factores independientes para X e Y

    Args:
        image: PIL Image
        scale_x: Factor de escala en X (1.0 = sin cambio)
        scale_y: Factor de escala en Y (1.0 = sin cambio)
        resample: Método de remuestreo (LANCZOS, BICUBIC, BILINEAR, NEAREST)

    Returns:
        PIL Image escalada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    # Validar factores de escala
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Los factores de escala deben ser positivos")

    width, height = image.size
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    # Asegurar dimensiones mínimas
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    return image.resize((new_width, new_height), resample)


def scale_uniform(image, scale_factor=1.0, resample=Image.LANCZOS):
    """
    Escalar imagen uniformemente (mismo factor para X e Y)

    Args:
        image: PIL Image
        scale_factor: Factor de escala uniforme
        resample: Método de remuestreo

    Returns:
        PIL Image escalada
    """
    return scale_image(image, scale_factor, scale_factor, resample)


def resize_to_dimensions(image, width=None, height=None, keep_aspect=True):
    """
    Redimensionar a dimensiones específicas

    Args:
        image: PIL Image
        width: Ancho deseado (None para calcular automáticamente)
        height: Alto deseado (None para calcular automáticamente)
        keep_aspect: Mantener proporción de aspecto

    Returns:
        PIL Image redimensionada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    orig_width, orig_height = image.size

    if width is None and height is None:
        return image.copy()

    if keep_aspect:
        if width is not None and height is None:
            # Calcular altura basado en ancho
            aspect_ratio = orig_height / orig_width
            height = int(width * aspect_ratio)
        elif height is not None and width is None:
            # Calcular ancho basado en altura
            aspect_ratio = orig_width / orig_height
            width = int(height * aspect_ratio)
        elif width is not None and height is not None:
            # Usar el factor más restrictivo
            width_ratio = width / orig_width
            height_ratio = height / orig_height
            ratio = min(width_ratio, height_ratio)
            width = int(orig_width * ratio)
            height = int(orig_height * ratio)
    else:
        if width is None:
            width = orig_width
        if height is None:
            height = orig_height

    return image.resize((width, height), Image.LANCZOS)


def scale_to_fit(image, max_width, max_height):
    """
    Escalar imagen para que quepa dentro de dimensiones máximas
    manteniendo la proporción de aspecto

    Args:
        image: PIL Image
        max_width: Ancho máximo
        max_height: Alto máximo

    Returns:
        PIL Image escalada
    """
    width, height = image.size

    # Si ya cabe, retornar copia
    if width <= max_width and height <= max_height:
        return image.copy()

    # Calcular ratio de escala
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale_ratio = min(width_ratio, height_ratio)

    new_width = int(width * scale_ratio)
    new_height = int(height * scale_ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)


def scale_percentage(image, percentage):
    """
    Escalar imagen por porcentaje

    Args:
        image: PIL Image
        percentage: Porcentaje (100 = sin cambio, 50 = mitad, 200 = doble)

    Returns:
        PIL Image escalada
    """
    if percentage <= 0:
        raise ValueError("El porcentaje debe ser positivo")

    scale_factor = percentage / 100.0
    return scale_uniform(image, scale_factor)