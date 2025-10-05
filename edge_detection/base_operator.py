"""
Clase base abstracta para operadores de detección de bordes
"""

from abc import ABC, abstractmethod
import numpy as np


class EdgeOperator(ABC):
    """Clase base abstracta para todos los operadores de detección de bordes"""

    def __init__(self):
        self.name = "Base Operator"
        self.description = ""

    @abstractmethod
    def apply(self, image_array):
        """
        Aplicar operador de detección de bordes

        Args:
            image_array: numpy array de la imagen en escala de grises

        Returns:
            tuple: (magnitude, angle, grad_x, grad_y)
                - magnitude: Magnitud del gradiente
                - angle: Ángulo del gradiente (puede ser None)
                - grad_x: Gradiente en X (puede ser None)
                - grad_y: Gradiente en Y (puede ser None)
        """
        pass

    def validate_input(self, image_array):
        """
        Validar entrada

        Args:
            image_array: numpy array

        Raises:
            ValueError: Si la imagen no es válida
        """
        if image_array is None:
            raise ValueError("La imagen no puede ser None")

        if not isinstance(image_array, np.ndarray):
            raise ValueError("La imagen debe ser un numpy array")

        if image_array.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D, dimensiones actuales: {image_array.ndim}")

        if image_array.size == 0:
            raise ValueError("La imagen está vacía")

    def get_name(self):
        """Obtener nombre del operador"""
        return self.name

    def get_description(self):
        """Obtener descripción del operador"""
        return self.description