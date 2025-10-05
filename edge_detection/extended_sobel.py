"""
Operador de Sobel Extendido
Basado en el libro - Sección 6.3.8
Figura 6.18: Máscaras 5x5, 7x7, 9x9, 11x11
"""

import numpy as np
from scipy import ndimage
from .base_operator import EdgeOperator
from .first_derivative import SobelOperator


class ExtendedSobelOperator(EdgeOperator):
    """
    Operador de Sobel Extendido
    Soporta tamaños 3x3, 5x5, 7x7, 9x9, 11x11
    Basado en Figura 6.18 del libro
    """

    def __init__(self, size=7):
        super().__init__()
        self.name = f"Sobel {size}x{size}"
        self.description = f"Sobel extendido con máscara {size}x{size}"
        self.size = size

        if size not in [3, 5, 7, 9, 11]:
            raise ValueError("Tamaño debe ser 3, 5, 7, 9 o 11")

    def apply(self, image_array):
        self.validate_input(image_array)

        if self.size == 3:
            # Usar Sobel estándar
            sobel = SobelOperator()
            return sobel.apply(image_array)

        # Obtener máscaras extendidas
        gx, gy = self._get_extended_masks(self.size)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def _get_extended_masks(self, size):
        """
        Obtener máscaras extendidas según tamaño
        Basadas en la Figura 6.18 del libro
        """
        if size == 5:
            # Máscara 5x5
            gx = np.array([
                [-1, -1, -2, -1, -1],
                [-1, -1, -2, -1, -1],
                [0, 0, 0, 0, 0],
                [1, 1, 2, 1, 1],
                [1, 1, 2, 1, 1]
            ], dtype=np.float64)
            gy = gx.T

        elif size == 7:
            # Máscara 7x7 (Figura 6.18)
            gx = np.array([
                [-1, -1, -1, -2, -1, -1, -1],
                [-1, -1, -1, -2, -1, -1, -1],
                [-1, -1, -1, -2, -1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1]
            ], dtype=np.float64)
            gy = gx.T

        elif size == 9:
            # Máscara 9x9
            gx = np.array([
                [-1, -1, -1, -1, -2, -1, -1, -1, -1],
                [-1, -1, -1, -1, -2, -1, -1, -1, -1],
                [-1, -1, -1, -1, -2, -1, -1, -1, -1],
                [-1, -1, -1, -1, -2, -1, -1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 2, 1, 1, 1, 1],
                [1, 1, 1, 1, 2, 1, 1, 1, 1],
                [1, 1, 1, 1, 2, 1, 1, 1, 1],
                [1, 1, 1, 1, 2, 1, 1, 1, 1]
            ], dtype=np.float64)
            gy = gx.T

        elif size == 11:
            # Máscara 11x11
            gx = np.array([
                [-1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
            ], dtype=np.float64)
            gy = gx.T

        else:
            raise ValueError(f"Tamaño {size} no soportado")

        return gx, gy

    def set_size(self, size):
        """Cambiar tamaño de la máscara"""
        if size not in [3, 5, 7, 9, 11]:
            raise ValueError("Tamaño debe ser 3, 5, 7, 9 o 11")
        self.size = size
        self.name = f"Sobel {size}x{size}"
        self.description = f"Sobel extendido con máscara {size}x{size}"