"""
Operador de Frei-Chen
Basado en el libro - Sección 6.3.7
Figura 6.15: 9 máscaras formando base vectorial completa
"""

import numpy as np
from scipy import ndimage
from .base_operator import EdgeOperator


class FreiChenOperator(EdgeOperator):
    """
    Máscaras de Frei-Chen (Figura 6.15 del libro)
    9 máscaras formando base vectorial completa

    Ecuación (6.10): R = suma(wi*zi) = ||w||||z||cos(θ) = w^t*z = (W,Z)
    Ecuación (6.13): cos(θ) = sqrt(M/S)
    donde M = suma((I_S, f_i)²) para i=1..4 (subespacio de bordes)
          S = suma((I_S, f_i)²) para i=1..9 (espacio completo)
    """

    def __init__(self):
        super().__init__()
        self.name = "Frei-Chen"
        self.description = "9 máscaras - base vectorial completa. Ec. (6.10), (6.13)"

    def apply(self, image_array):
        self.validate_input(image_array)

        sqrt2 = np.sqrt(2)

        # Las 9 máscaras EXACTAS de la Figura 6.15
        masks = [
            # Bordes (f1-f4) - Subespacio de bordes
            (1 / (2 * sqrt2)) * np.array([[1, sqrt2, 1],
                                          [0, 0, 0],
                                          [-1, -sqrt2, -1]]),

            (1 / (2 * sqrt2)) * np.array([[1, 0, -1],
                                          [sqrt2, 0, -sqrt2],
                                          [1, 0, -1]]),

            (1 / (2 * sqrt2)) * np.array([[0, -1, sqrt2],
                                          [1, 0, -1],
                                          [-sqrt2, 1, 0]]),

            (1 / (2 * sqrt2)) * np.array([[sqrt2, -1, 0],
                                          [-1, 0, 1],
                                          [0, 1, -sqrt2]]),

            # Líneas (f5-f6) - Subespacio de líneas
            (1 / 2) * np.array([[0, 1, 0],
                                [-1, 0, -1],
                                [0, 1, 0]]),

            (1 / 2) * np.array([[-1, 0, 1],
                                [0, 0, 0],
                                [1, 0, -1]]),

            # Promedio (f7-f8) - Subespacio promedio
            (1 / 6) * np.array([[1, -2, 1],
                                [-2, 4, -2],
                                [1, -2, 1]]),

            (1 / 6) * np.array([[-2, 1, -2],
                                [1, 4, 1],
                                [-2, 1, -2]]),

            # Isotrópico (f9) - Promedio simple
            (1 / 3) * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])
        ]

        # Proyecciones (I_S, f_i) según ecuación (6.10)
        projections = []
        for mask in masks:
            projection = ndimage.convolve(
                image_array.astype(np.float64),
                mask.astype(np.float64)
            )
            projections.append(projection)

        # Subespacio de bordes: solo primeras 4 máscaras (f1-f4)
        # Ecuación (6.13): M = suma((I_S, f_i)²) para i=1..4
        #                  S = suma((I_S, f_i)²) para i=1..9
        edge_responses = [p ** 2 for p in projections[:4]]
        all_responses = [p ** 2 for p in projections]

        M = np.sum(edge_responses, axis=0)
        S = np.sum(all_responses, axis=0)

        # Evitar división por cero
        S_safe = np.where(S > 0, S, 1)

        # cos(θ) = sqrt(M/S) del libro (Ecuación 6.13)
        cos_theta = np.sqrt(M / S_safe)

        # Magnitud de bordes
        magnitude = np.sqrt(M)

        # Retornar magnitude y cos_theta
        # El ángulo aquí es cos(θ), no θ en radianes
        return magnitude, cos_theta, None, None

    def get_edge_strength(self, image_array):
        """
        Obtener fuerza de borde normalizada

        Returns:
            numpy array con valores entre 0 y 1 indicando fuerza de borde
        """
        magnitude, cos_theta, _, _ = self.apply(image_array)
        return cos_theta  # Ya está entre 0 y 1