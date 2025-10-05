"""
Operadores de primera derivada
Basados en el libro 'Visión por Computador: Imágenes Digitales y Aplicaciones'
Sección 6.3.1 - 6.3.4
"""

import numpy as np
from scipy import ndimage
from .base_operator import EdgeOperator


class GradientOperator(EdgeOperator):
    """
    Gradiente Básico - Diferencias finitas (PDF pág 2)
    Gx = [f(x+1,y) - f(x-1,y)] / 2
    Gy = [f(x,y+1) - f(x,y-1)] / 2
    """

    def __init__(self):
        super().__init__()
        self.name = "Gradiente Básico"
        self.description = "Diferencias finitas. Rápido pero sensible al ruido."

    def apply(self, image_array):
        self.validate_input(image_array)

        grad_x = np.zeros_like(image_array, dtype=np.float64)
        grad_y = np.zeros_like(image_array, dtype=np.float64)

        # Diferencias finitas centradas
        grad_x[:, 1:-1] = (image_array[:, 2:] - image_array[:, :-2]) / 2.0
        grad_y[1:-1, :] = (image_array[2:, :] - image_array[:-2, :]) / 2.0

        # Bordes
        grad_x[:, 0] = image_array[:, 1] - image_array[:, 0]
        grad_x[:, -1] = image_array[:, -1] - image_array[:, -2]
        grad_y[0, :] = image_array[1, :] - image_array[0, :]
        grad_y[-1, :] = image_array[-1, :] - image_array[-2, :]

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y


class SobelOperator(EdgeOperator):
    """
    Operador de Sobel (PDF pág 5-7)
    Figura 6.4: Máscaras exactas
    Ecuación (6.6): Gx = (z3 + 2z6 + z9) - (z1 + 2z4 + z7)
                    Gy = (z1 + 2z2 + z3) - (z7 + 2z8 + z9)
    """

    def __init__(self):
        super().__init__()
        self.name = "Sobel"
        self.description = "Balance óptimo entre precisión y suavizado. Ec. (6.6)"

    def apply(self, image_array):
        self.validate_input(image_array)

        # Máscaras exactas del PDF (Figura 6.4b y 6.4c)
        gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)

        gy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        # Ecuación (6.2) y (6.3) del PDF
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y


class PrewittOperator(EdgeOperator):
    """
    Operador de Prewitt (PDF pág 7-8)
    Figura 6.7: Máscaras exactas
    Similar a Sobel pero con coeficientes uniformes
    """

    def __init__(self):
        super().__init__()
        self.name = "Prewitt"
        self.description = "Similar a Sobel con coeficientes uniformes."

    def apply(self, image_array):
        self.validate_input(image_array)

        # Máscaras del PDF (Figura 6.7)
        gx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float64)

        gy = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y


class RobertsOperator(EdgeOperator):
    """
    Operador de Roberts (PDF pág 8-9)
    Figura 6.9: Máscaras 2x2
    Ecuación (6.7): D1 = f(x,y) - f(x-1,y-1)
                    D2 = f(x,y-1) - f(x-1,y)
    Ecuación (6.8):
        Forma 1: R = sqrt(D1² + D2²)
        Forma 2: R = |D1| + |D2|
    """

    def __init__(self, form="sqrt"):
        super().__init__()
        self.name = "Roberts"
        self.description = "Operador diagonal 2x2. Ec. (6.7) y (6.8)"
        self.form = form  # "sqrt" o "abs"

    def apply(self, image_array):
        self.validate_input(image_array)

        # Máscaras 2x2 del libro expandidas a 3x3 para convolución
        gx = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 0]], dtype=np.float64)

        gy = np.array([[0, 1, 0],
                       [-1, 0, 0],
                       [0, 0, 0]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        # Seleccionar forma según ecuación (6.8)
        if self.form == "sqrt":
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        else:  # "abs"
            magnitude = np.abs(grad_x) + np.abs(grad_y)

        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def set_form(self, form):
        """Cambiar forma de cálculo"""
        if form in ["sqrt", "abs"]:
            self.form = form
        else:
            raise ValueError("Forma debe ser 'sqrt' o 'abs'")