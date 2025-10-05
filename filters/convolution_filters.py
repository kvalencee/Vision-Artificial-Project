"""
Filtros basados en convolución
"""

from PIL import Image, ImageFilter


def apply_blur(image, radius=2):
    """
    Aplicar desenfoque gaussiano

    Args:
        image: PIL Image
        radius: Radio del desenfoque

    Returns:
        PIL Image desenfocada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_box_blur(image, radius=2):
    """
    Aplicar desenfoque de caja (promedio)

    Args:
        image: PIL Image
        radius: Radio del desenfoque

    Returns:
        PIL Image desenfocada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.BoxBlur(radius=radius))


def apply_detail(image):
    """
    Realzar detalles de la imagen

    Args:
        image: PIL Image

    Returns:
        PIL Image con detalles realzados
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.DETAIL)


def apply_edges(image):
    """
    Detectar bordes básico

    Args:
        image: PIL Image

    Returns:
        PIL Image con bordes detectados
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.FIND_EDGES)


def apply_enhance(image):
    """
    Realzar bordes de la imagen

    Args:
        image: PIL Image

    Returns:
        PIL Image con bordes realzados
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.EDGE_ENHANCE)


def apply_enhance_more(image):
    """
    Realzar bordes más fuertemente

    Args:
        image: PIL Image

    Returns:
        PIL Image con bordes muy realzados
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.EDGE_ENHANCE_MORE)


def apply_emboss(image):
    """
    Aplicar efecto relieve (emboss)

    Args:
        image: PIL Image

    Returns:
        PIL Image con efecto relieve
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.EMBOSS)


def apply_contour(image):
    """
    Aplicar detección de contornos

    Args:
        image: PIL Image

    Returns:
        PIL Image con contornos
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.CONTOUR)


def apply_smooth(image):
    """
    Aplicar suavizado

    Args:
        image: PIL Image

    Returns:
        PIL Image suavizada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.SMOOTH)


def apply_smooth_more(image):
    """
    Aplicar suavizado más fuerte

    Args:
        image: PIL Image

    Returns:
        PIL Image muy suavizada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.SMOOTH_MORE)


def apply_sharpen(image):
    """
    Aplicar filtro de enfoque

    Args:
        image: PIL Image

    Returns:
        PIL Image enfocada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.SHARPEN)


def apply_median_filter(image, size=3):
    """
    Aplicar filtro de mediana (reduce ruido sal y pimienta)

    Args:
        image: PIL Image
        size: Tamaño del kernel (debe ser impar)

    Returns:
        PIL Image filtrada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.MedianFilter(size=size))


def apply_min_filter(image, size=3):
    """
    Aplicar filtro mínimo (erosión)

    Args:
        image: PIL Image
        size: Tamaño del kernel

    Returns:
        PIL Image filtrada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.MinFilter(size=size))


def apply_max_filter(image, size=3):
    """
    Aplicar filtro máximo (dilatación)

    Args:
        image: PIL Image
        size: Tamaño del kernel

    Returns:
        PIL Image filtrada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.MaxFilter(size=size))


def apply_mode_filter(image, size=3):
    """
    Aplicar filtro de moda

    Args:
        image: PIL Image
        size: Tamaño del kernel

    Returns:
        PIL Image filtrada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(ImageFilter.ModeFilter(size=size))