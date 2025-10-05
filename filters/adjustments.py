"""
Ajustes de imagen (brillo, contraste, saturación, nitidez)
"""

from PIL import Image, ImageEnhance, ImageFilter


def adjust_brightness(image, factor=1.0):
    """
    Ajustar brillo de la imagen

    Args:
        image: PIL Image
        factor: Factor de brillo (1.0 = sin cambio, >1 más brillante, <1 más oscuro)

    Returns:
        PIL Image con brillo ajustado
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    if factor < 0:
        factor = 0

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor=1.0):
    """
    Ajustar contraste de la imagen

    Args:
        image: PIL Image
        factor: Factor de contraste (1.0 = sin cambio, >1 más contraste, <1 menos)

    Returns:
        PIL Image con contraste ajustado
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    if factor < 0:
        factor = 0

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_saturation(image, factor=1.0):
    """
    Ajustar saturación de la imagen

    Args:
        image: PIL Image
        factor: Factor de saturación (1.0 = sin cambio, >1 más saturado, 0 = grises)

    Returns:
        PIL Image con saturación ajustada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    if factor < 0:
        factor = 0

    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def adjust_sharpness(image, factor=1.0):
    """
    Ajustar nitidez de la imagen

    Args:
        image: PIL Image
        factor: Factor de nitidez (1.0 = sin cambio, >1 más nítido, <1 más suave)

    Returns:
        PIL Image con nitidez ajustada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    if factor < 0:
        factor = 0

    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def apply_sharpen(image, radius=2, percent=150, threshold=3):
    """
    Aplicar máscara de enfoque (unsharp mask)

    Args:
        image: PIL Image
        radius: Radio del desenfoque gaussiano
        percent: Porcentaje de nitidez (100 = normal, >100 más nítido)
        threshold: Umbral mínimo de diferencia

    Returns:
        PIL Image con nitidez mejorada
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    return image.filter(
        ImageFilter.UnsharpMask(
            radius=radius,
            percent=percent,
            threshold=threshold
        )
    )


def adjust_gamma(image, gamma=1.0):
    """
    Ajustar gamma de la imagen

    Args:
        image: PIL Image
        gamma: Valor gamma (1.0 = sin cambio, <1 más claro, >1 más oscuro)

    Returns:
        PIL Image con gamma ajustado
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    import numpy as np

    img_array = np.array(image).astype(np.float32) / 255.0

    # Aplicar corrección gamma
    corrected = np.power(img_array, 1.0 / gamma)
    corrected = (corrected * 255).astype(np.uint8)

    return Image.fromarray(corrected)


def auto_contrast(image, cutoff=0):
    """
    Ajustar contraste automáticamente

    Args:
        image: PIL Image
        cutoff: Porcentaje de píxeles a ignorar en extremos

    Returns:
        PIL Image con contraste automático
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    from PIL import ImageOps
    return ImageOps.autocontrast(image, cutoff=cutoff)


def equalize_histogram(image):
    """
    Ecualizar histograma de la imagen

    Args:
        image: PIL Image

    Returns:
        PIL Image con histograma ecualizado
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    from PIL import ImageOps
    return ImageOps.equalize(image)


def apply_all_adjustments(image, brightness=1.0, contrast=1.0,
                          saturation=1.0, sharpen_factor=1.0):
    """
    Aplicar todos los ajustes en secuencia

    Args:
        image: PIL Image
        brightness: Factor de brillo
        contrast: Factor de contraste
        saturation: Factor de saturación
        sharpen_factor: Factor de nitidez

    Returns:
        PIL Image con todos los ajustes aplicados
    """
    if not isinstance(image, Image.Image):
        raise TypeError("El parámetro debe ser una imagen PIL")

    result = image.copy()

    # Aplicar en orden óptimo
    if brightness != 1.0:
        result = adjust_brightness(result, brightness)

    if contrast != 1.0:
        result = adjust_contrast(result, contrast)

    if saturation != 1.0:
        result = adjust_saturation(result, saturation)

    if sharpen_factor > 1.0:
        result = apply_sharpen(result, radius=2, percent=int(150 * sharpen_factor))

    return result


def normalize_levels(image):
    """
    Normalizar niveles de la imagen

    Args:
        image: PIL Image

    Returns:
        PIL Image normalizada
    """
    import numpy as np

    img_array = np.array(image).astype(np.float32)

    # Normalizar cada canal
    for i in range(img_array.shape[2] if len(img_array.shape) > 2 else 1):
        if len(img_array.shape) > 2:
            channel = img_array[:, :, i]
        else:
            channel = img_array

        min_val = channel.min()
        max_val = channel.max()

        if max_val > min_val:
            normalized = (channel - min_val) / (max_val - min_val) * 255
            if len(img_array.shape) > 2:
                img_array[:, :, i] = normalized
            else:
                img_array = normalized

    return Image.fromarray(img_array.astype(np.uint8))