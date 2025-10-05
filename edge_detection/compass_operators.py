"""
Operadores Compass (Kirsch y Robinson)
Basados en el libro - Sección 6.3.5 y 6.3.6
"""

import numpy as np
from scipy import ndimage
from .base_operator import EdgeOperator


class KirschOperator(EdgeOperator):
    """
    Máscaras de Kirsch (Figura 6.11 del libro)
    8 máscaras direccionales (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    R = max{k0, k1, ..., k7}
    """

    def __init__(self):
        super().__init__()
        self.name = "Kirsch"
        self.description = "8 máscaras direccionales. Toma máximo. Fig. 6.11"

    def apply(self, image_array):
        self.validate_input(image_array)

        # Máscaras exactas de la Figura 6.11 del libro
        # k0 a k7 en orden de rotación
        masks = [
            np.array([[5, 5, 5],
                      [-3, 0, -3],
                      [-3, -3, -3]]),  # 0° (k0)

            np.array([[5, 5, -3],
                      [5, 0, -3],
                      [-3, -3, -3]]),  # 45° (k1)

            np.array([[5, -3, -3],
                      [5, 0, -3],
                      [5, -3, -3]]),  # 90° (k2)

            np.array([[-3, -3, -3],
                      [5, 0, -3],
                      [5, 5, -3]]),  # 135° (k3)

            np.array([[-3, -3, -3],
                      [-3, 0, -3],
                      [5, 5, 5]]),  # 180° (k4)

            np.array([[-3, -3, -3],
                      [-3, 0, 5],
                      [-3, 5, 5]]),  # 225° (k5)

            np.array([[-3, -3, 5],
                      [-3, 0, 5],
                      [-3, -3, 5]]),  # 270° (k6)

            np.array([[-3, 5, 5],
                      [-3, 0, 5],
                      [-3, -3, -3]])  # 315° (k7)
        ]

        responses = []
        for mask in masks:
            response = ndimage.convolve(
                image_array.astype(np.float64),
                mask.astype(np.float64)
            )
            responses.append(response)

        # Máximo en cada posición (según libro pág 141)
        magnitude = np.maximum.reduce(responses)

        # Ángulo correspondiente al máximo
        responses_stack = np.stack(responses, axis=-1)
        angle_indices = np.argmax(responses_stack, axis=-1)
        angle = angle_indices * 45.0  # Convertir índice a grados

        return magnitude, np.deg2rad(angle), None, None


class RobinsonOperator(EdgeOperator):
    """
    Máscaras de Robinson (Figura 6.13 del libro)
    Máscara inicial r0 y 7 rotaciones

    Figura 6.13: r0 = [[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]
    """

    def __init__(self):
        super().__init__()
        self.name = "Robinson"
        self.description = "Similar a Kirsch, máscara inicial diferente. Fig. 6.13"

    def apply(self, image_array):
        self.validate_input(image_array)

        # Las 8 máscaras generadas mediante rotación de 45°
        # Basadas en la máscara r0 de la Figura 6.13
        masks = [
            np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]]),  # 0°

            np.array([[0, 1, 2],
                      [-1, 0, 1],
                      [-2, -1, 0]]),  # 45°

            np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]),  # 90°

            np.array([[2, 1, 0],
                      [1, 0, -1],
                      [0, -1, -2]]),  # 135°

            np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]),  # 180°

            np.array([[0, -1, -2],
                      [1, 0, -1],
                      [2, 1, 0]]),  # 225°

            np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]]),  # 270°

            np.array([[-2, -1, 0],
                      [-1, 0, 1],
                      [0, 1, 2]])  # 315°
        ]

        responses = []
        for mask in masks:
            response = ndimage.convolve(
                image_array.astype(np.float64),
                mask.astype(np.float64)
            )
            responses.append(response)

        magnitude = np.maximum.reduce(responses)
        responses_stack = np.stack(responses, axis=-1)
        angle_indices = np.argmax(responses_stack, axis=-1)
        angle = angle_indices * 45.0

        return magnitude, np.deg2rad(angle), None, None