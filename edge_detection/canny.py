"""
Algoritmo de Canny
Basado en el libro - Sección 6.3.9
3 módulos principales:
1. Obtención del gradiente (magnitud y ángulo)
2. Adelgazamiento (supresión no máxima)
3. Histéresis de umbral
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, binary_dilation
from .base_operator import EdgeOperator
from .first_derivative import SobelOperator


class CannyOperator(EdgeOperator):
    """
    Algoritmo de Canny completo

    Parámetros:
    - sigma: Desviación estándar del filtro gaussiano
    - low_threshold (t1): Umbral bajo para histéresis
    - high_threshold (t2): Umbral alto para histéresis
    """

    def __init__(self, sigma=1.0, low_threshold=50, high_threshold=150):
        super().__init__()
        self.name = "Canny"
        self.description = "Algoritmo óptimo. 3 módulos: gradiente, supresión no máxima, histéresis."
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, image_array):
        self.validate_input(image_array)

        try:
            # Intentar usar OpenCV (más eficiente)
            edges = self._apply_opencv(image_array)
            return edges, None, None, None
        except:
            # Implementación manual según el libro
            return self._apply_manual(image_array)

    def _apply_opencv(self, image_array):
        """Implementación usando OpenCV"""
        img_uint8 = image_array.astype(np.uint8)

        # Calcular tamaño del kernel gaussiano
        ksize = int(2 * np.ceil(2 * self.sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1

        # Aplicar desenfoque gaussiano
        blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), self.sigma)

        # Aplicar Canny
        edges = cv2.Canny(
            blurred,
            int(self.low_threshold),
            int(self.high_threshold)
        )

        return edges

    def _apply_manual(self, image_array):
        """Implementación manual según el libro"""

        # 1) Obtención del gradiente (PDF pág 16)
        # a) Suavizar con núcleo Gaussiano
        smoothed = gaussian_filter(
            image_array.astype(np.float64),
            sigma=self.sigma
        )

        # b) Calcular magnitud y ángulo del gradiente (ecuación 6.2 y 6.3)
        sobel = SobelOperator()
        magnitude, angle, grad_x, grad_y = sobel.apply(smoothed)

        # 2) Supresión no máxima (PDF pág 16)
        suppressed = self._non_maximum_suppression(magnitude, angle)

        # 3) Histéresis de umbral (PDF pág 16-18)
        edges = self._hysteresis_threshold(
            suppressed,
            self.low_threshold,
            self.high_threshold
        )

        return (edges * 255).astype(np.uint8), angle, grad_x, grad_y

    def _non_maximum_suppression(self, magnitude, angle):
        """
        Adelgazamiento de bordes - Supresión no máxima

        Para cada píxel, verificar si es máximo local en la dirección
        perpendicular al borde (dirección del gradiente)
        """
        rows, cols = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # Convertir ángulos a grados y normalizar a [0, 180)
        angle_deg = (np.rad2deg(angle) + 180) % 180

        # 4 direcciones: 0°, 45°, 90°, 135° (PDF pág 16)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                current = magnitude[i, j]
                ang = angle_deg[i, j]

                # Determinar dirección que mejor aproxima E_θ(i,j)
                if (ang >= 0 and ang < 22.5) or (ang >= 157.5 and ang < 180):
                    # Dirección horizontal (0°)
                    neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
                elif ang >= 22.5 and ang < 67.5:
                    # Dirección diagonal (45°)
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
                elif ang >= 67.5 and ang < 112.5:
                    # Dirección vertical (90°)
                    neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
                else:  # 112.5 - 157.5
                    # Dirección diagonal (135°)
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

                # Si E_M(i,j) es máximo local, mantener; sino suprimir
                if current >= max(neighbors):
                    suppressed[i, j] = current

        return suppressed

    def _hysteresis_threshold(self, image, low_threshold, high_threshold):
        """
        Histéresis de umbral (PDF pág 16-18)
        Algoritmo con dos umbrales t1 y t2 (t1 < t2)

        a) Tomar I_N como entrada y dos umbrales t1 y t2
        b) Para todos los puntos de I_N:
           b.1) Localizar siguiente punto de borde no explorado I_N(i,j)
           b.2) Seguir cadenas de máximos locales conectados
        c) Salida es conjunto de bordes conectados
        """
        # Bordes fuertes: por encima del umbral alto
        strong_edges = image > high_threshold

        # Bordes débiles: entre umbral bajo y alto
        weak_edges = (image >= low_threshold) & (image <= high_threshold)

        # Dilatar bordes fuertes para conectar con débiles adyacentes
        strong_dilated = binary_dilation(strong_edges, structure=np.ones((3, 3)))

        # Mantener bordes débiles conectados a bordes fuertes
        connected_weak = weak_edges & strong_dilated

        # Resultado final: bordes fuertes + bordes débiles conectados
        edges = strong_edges | connected_weak

        return edges.astype(np.float32)

    def set_parameters(self, sigma=None, low_threshold=None, high_threshold=None):
        """Actualizar parámetros del algoritmo"""
        if sigma is not None:
            self.sigma = sigma
        if low_threshold is not None:
            self.low_threshold = low_threshold
        if high_threshold is not None:
            self.high_threshold = high_threshold