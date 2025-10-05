"""
Operadores de Segunda Derivada
Basado en el libro - Laplaciano y detección de zero-crossings
"""

import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from .base_operator import EdgeOperator


class LaplacianOperator(EdgeOperator):
    """
    Laplaciano de la Gaussiana (LoG)

    Segunda derivada - detecta zero-crossings (PDF pág 2-3)
    La segunda derivada es cero en el inicio y final de una transición
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.name = "Laplaciano"
        self.description = "Segunda derivada. Detecta zero-crossings."
        self.sigma = sigma

    def apply(self, image_array):
        self.validate_input(image_array)

        # Suavizado gaussiano
        smoothed = gaussian_filter(
            image_array.astype(np.float64),
            sigma=self.sigma
        )

        # Laplaciano (segunda derivada)
        laplacian = laplace(smoothed)

        # Detectar zero-crossings
        zero_crossings = self._detect_zero_crossings(laplacian)

        return zero_crossings, None, None, None

    def _detect_zero_crossings(self, laplacian):
        """
        Detectar cambios de signo (zero-crossings) - PDF pág 2

        La segunda derivada es cero en el inicio y final de una transición.
        Un zero-crossing ocurre cuando hay cambio de signo entre píxeles vecinos.
        """
        zc = np.zeros_like(laplacian, dtype=np.uint8)

        # Verificar cambios de signo en 4 direcciones principales
        # Horizontal, Vertical, Diagonal principal, Diagonal secundaria
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # Desplazar imagen
            rolled = np.roll(np.roll(laplacian, dx, axis=0), dy, axis=1)

            # Detectar donde el producto es negativo (cambio de signo)
            zc |= ((laplacian * rolled) < 0).astype(np.uint8)

        return zc * 255

    def set_sigma(self, sigma):
        """Actualizar valor de sigma"""
        if sigma > 0:
            self.sigma = sigma
        else:
            raise ValueError("Sigma debe ser positivo")


class LoGOperator(EdgeOperator):
    """
    Laplacian of Gaussian (LoG) - Versión alternativa
    Combina suavizado gaussiano y laplaciano en una sola operación
    """

    def __init__(self, sigma=1.0, threshold=0):
        super().__init__()
        self.name = "LoG"
        self.description = "Laplaciano de Gaussiana con umbral"
        self.sigma = sigma
        self.threshold = threshold

    def apply(self, image_array):
        self.validate_input(image_array)

        # Suavizado gaussiano
        smoothed = gaussian_filter(
            image_array.astype(np.float64),
            sigma=self.sigma
        )

        # Aplicar Laplaciano
        laplacian = laplace(smoothed)

        # Detectar zero-crossings con umbral
        zero_crossings = self._detect_zero_crossings_threshold(
            laplacian,
            self.threshold
        )

        return zero_crossings, None, None, None

    def _detect_zero_crossings_threshold(self, laplacian, threshold):
        """
        Detectar zero-crossings con umbral mínimo

        Solo detecta zero-crossings donde la magnitud del cambio
        excede el umbral especificado
        """
        zc = np.zeros_like(laplacian, dtype=np.uint8)

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            rolled = np.roll(np.roll(laplacian, dx, axis=0), dy, axis=1)

            # Cambio de signo Y magnitud suficiente
            sign_change = (laplacian * rolled) < 0
            magnitude = np.abs(laplacian - rolled)

            zc |= (sign_change & (magnitude > threshold)).astype(np.uint8)

        return zc * 255

    def set_parameters(self, sigma=None, threshold=None):
        """Actualizar parámetros"""
        if sigma is not None and sigma > 0:
            self.sigma = sigma
        if threshold is not None:
            self.threshold = threshold