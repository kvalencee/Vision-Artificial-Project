"""
Gestor de imágenes
Maneja carga, guardado y manipulación de imágenes
"""

import os
from PIL import Image
import numpy as np


class ImageManager:
    """Gestor centralizado de operaciones de imagen"""

    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.filename = None
        self.zoom_factor = 1.0

    def load_image(self, filepath):
        """
        Cargar imagen desde archivo

        Args:
            filepath: Ruta del archivo

        Returns:
            PIL Image cargada

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo no es una imagen válida
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        try:
            image = Image.open(filepath).convert('RGB')
            self.original_image = image.copy()
            self.current_image = image.copy()
            self.filename = os.path.basename(filepath)
            self.zoom_factor = 1.0

            return self.current_image
        except Exception as e:
            raise ValueError(f"Error al cargar imagen: {str(e)}")

    def save_image(self, filepath, image=None):
        """
        Guardar imagen en archivo

        Args:
            filepath: Ruta donde guardar
            image: PIL Image a guardar (None usa current_image)

        Raises:
            ValueError: Si no hay imagen para guardar
        """
        img_to_save = image if image is not None else self.current_image

        if img_to_save is None:
            raise ValueError("No hay imagen para guardar")

        try:
            img_to_save.save(filepath)
        except Exception as e:
            raise ValueError(f"Error al guardar imagen: {str(e)}")

    def get_current_image(self):
        """Obtener imagen actual"""
        return self.current_image

    def get_original_image(self):
        """Obtener imagen original"""
        return self.original_image

    def set_current_image(self, image):
        """
        Establecer imagen actual

        Args:
            image: PIL Image
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Debe ser una imagen PIL")
        self.current_image = image

    def reset_to_original(self):
        """Restaurar a imagen original"""
        if self.original_image is None:
            raise ValueError("No hay imagen original cargada")
        self.current_image = self.original_image.copy()
        self.zoom_factor = 1.0

    def get_image_info(self):
        """
        Obtener información de la imagen actual

        Returns:
            dict con información de la imagen
        """
        if self.current_image is None:
            return {
                'filename': 'N/A',
                'width': 0,
                'height': 0,
                'mode': 'N/A',
                'format': 'N/A'
            }

        width, height = self.current_image.size

        return {
            'filename': self.filename or 'N/A',
            'width': width,
            'height': height,
            'mode': self.current_image.mode,
            'format': self.current_image.format or 'N/A'
        }

    def get_image_array(self, grayscale=False):
        """
        Obtener imagen como numpy array

        Args:
            grayscale: Si True, convierte a escala de grises

        Returns:
            numpy array
        """
        if self.current_image is None:
            return None

        if grayscale:
            gray_image = self.current_image.convert('L')
            return np.array(gray_image, dtype=np.float32)

        return np.array(self.current_image)

    def set_zoom(self, factor):
        """
        Establecer factor de zoom

        Args:
            factor: Factor de zoom (>0)
        """
        if factor <= 0:
            raise ValueError("El factor de zoom debe ser positivo")
        self.zoom_factor = factor

    def get_zoom(self):
        """Obtener factor de zoom actual"""
        return self.zoom_factor

    def has_image(self):
        """Verificar si hay imagen cargada"""
        return self.current_image is not None

    def get_size(self):
        """Obtener dimensiones de la imagen actual"""
        if self.current_image is None:
            return (0, 0)
        return self.current_image.size

    def copy_current(self):
        """Obtener copia de la imagen actual"""
        if self.current_image is None:
            return None
        return self.current_image.copy()