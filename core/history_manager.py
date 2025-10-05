"""
Gestor de historial para deshacer/rehacer
"""

from PIL import Image
import sys

from utils.constants import MAX_HISTORY_SIZE


class HistoryManager:
    """Gestor de historial con funcionalidad de deshacer/rehacer"""

    def __init__(self, max_size=MAX_HISTORY_SIZE):
        self.history = []
        self.current_index = -1
        self.max_size = max_size

    def add(self, image):
        """
        Añadir imagen al historial

        Args:
            image: PIL Image a añadir
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Debe ser una imagen PIL")

        # Si estamos en medio del historial, eliminar elementos posteriores
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]

        # Añadir nueva imagen
        self.history.append(image.copy())
        self.current_index = len(self.history) - 1

        # Limitar tamaño del historial
        if len(self.history) > self.max_size:
            self.history.pop(0)
            self.current_index -= 1

    def undo(self):
        """
        Deshacer - volver a imagen anterior

        Returns:
            PIL Image anterior o None si no se puede deshacer
        """
        if not self.can_undo():
            return None

        self.current_index -= 1
        return self.history[self.current_index].copy()

    def redo(self):
        """
        Rehacer - ir a imagen siguiente

        Returns:
            PIL Image siguiente o None si no se puede rehacer
        """
        if not self.can_redo():
            return None

        self.current_index += 1
        return self.history[self.current_index].copy()

    def can_undo(self):
        """Verificar si se puede deshacer"""
        return self.current_index > 0

    def can_redo(self):
        """Verificar si se puede rehacer"""
        return self.current_index < len(self.history) - 1

    def get_current(self):
        """
        Obtener imagen actual del historial

        Returns:
            PIL Image actual o None si historial vacío
        """
        if self.current_index < 0 or self.current_index >= len(self.history):
            return None
        return self.history[self.current_index].copy()

    def clear(self):
        """Limpiar historial"""
        self.history = []
        self.current_index = -1

    def get_info(self):
        """
        Obtener información del historial

        Returns:
            dict con información del historial
        """
        return {
            'total': len(self.history),
            'current': self.current_index + 1,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo()
        }

    def initialize_with_image(self, image):
        """
        Inicializar historial con una imagen

        Args:
            image: PIL Image inicial
        """
        self.clear()
        self.add(image)

    def get_history_size(self):
        """Obtener tamaño del historial"""
        return len(self.history)

    def get_current_index(self):
        """Obtener índice actual"""
        return self.current_index

    def set_max_size(self, max_size):
        """
        Cambiar tamaño máximo del historial

        Args:
            max_size: Nuevo tamaño máximo
        """
        if max_size < 1:
            raise ValueError("El tamaño máximo debe ser al menos 1")

        self.max_size = max_size

        # Ajustar historial si es necesario
        if len(self.history) > max_size:
            # Mantener las más recientes
            removed = len(self.history) - max_size
            self.history = self.history[removed:]
            self.current_index = max(0, self.current_index - removed)