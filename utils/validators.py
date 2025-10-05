"""
Validadores para parámetros y datos
"""

import numpy as np
from .constants import MAX_IMAGE_SIZE


def validate_image_array(image_array):
    """
    Validar que el array de imagen sea válido

    Args:
        image_array: numpy array

    Returns:
        tuple: (is_valid, error_message)
    """
    if image_array is None:
        return False, "Array de imagen es None"

    if not isinstance(image_array, np.ndarray):
        return False, "No es un numpy array"

    if image_array.size == 0:
        return False, "Array de imagen está vacío"

    if image_array.ndim not in [2, 3]:
        return False, f"Dimensiones incorrectas: {image_array.ndim}. Debe ser 2D o 3D"

    return True, ""


def validate_image_size(image_array, max_size=MAX_IMAGE_SIZE):
    """
    Validar que la imagen no sea demasiado grande

    Args:
        image_array: numpy array
        max_size: Tamaño máximo permitido en píxeles

    Returns:
        tuple: (is_valid, warning_message)
    """
    if image_array.size > max_size:
        return False, f"Imagen muy grande ({image_array.size} píxeles). Puede tardar varios segundos."
    return True, ""


def validate_threshold(value, min_val=0, max_val=255):
    """
    Validar valor de umbral

    Args:
        value: Valor a validar
        min_val: Valor mínimo
        max_val: Valor máximo

    Returns:
        bool: True si es válido
    """
    try:
        val = float(value)
        return min_val <= val <= max_val
    except (ValueError, TypeError):
        return False


def validate_sigma(value, min_val=0.5, max_val=5.0):
    """
    Validar valor de sigma para filtro gaussiano

    Args:
        value: Valor a validar
        min_val: Valor mínimo
        max_val: Valor máximo

    Returns:
        bool: True si es válido
    """
    try:
        val = float(value)
        return min_val <= val <= max_val
    except (ValueError, TypeError):
        return False


def validate_mask_size(size):
    """
    Validar tamaño de máscara

    Args:
        size: Tamaño de la máscara

    Returns:
        tuple: (is_valid, error_message)
    """
    from .constants import EXTENDED_MASK_SIZES

    if size not in EXTENDED_MASK_SIZES:
        return False, f"Tamaño {size} no soportado. Use: {EXTENDED_MASK_SIZES}"

    return True, ""


def validate_roberts_form(form):
    """
    Validar forma del operador Roberts

    Args:
        form: Forma ("sqrt" o "abs")

    Returns:
        bool: True si es válido
    """
    return form in ["sqrt", "abs"]


def validate_operator(operator):
    """
    Validar que el operador existe

    Args:
        operator: Nombre del operador

    Returns:
        tuple: (is_valid, error_message)
    """
    from .constants import EDGE_OPERATORS

    if operator not in EDGE_OPERATORS:
        return False, f"Operador '{operator}' no reconocido"

    return True, ""


def sanitize_float(value, default=1.0, min_val=None, max_val=None):
    """
    Sanitizar valor float con límites opcionales

    Args:
        value: Valor a sanitizar
        default: Valor por defecto si falla
        min_val: Valor mínimo opcional
        max_val: Valor máximo opcional

    Returns:
        float: Valor sanitizado
    """
    try:
        val = float(value)
        if min_val is not None:
            val = max(val, min_val)
        if max_val is not None:
            val = min(val, max_val)
        return val
    except (ValueError, TypeError):
        return default


def sanitize_int(value, default=1, min_val=None, max_val=None):
    """
    Sanitizar valor int con límites opcionales

    Args:
        value: Valor a sanitizar
        default: Valor por defecto si falla
        min_val: Valor mínimo opcional
        max_val: Valor máximo opcional

    Returns:
        int: Valor sanitizado
    """
    try:
        val = int(value)
        if min_val is not None:
            val = max(val, min_val)
        if max_val is not None:
            val = min(val, max_val)
        return val
    except (ValueError, TypeError):
        return default