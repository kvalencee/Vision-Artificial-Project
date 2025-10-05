"""
PhotoEscom - Editor de Fotos Profesional
Punto de entrada principal de la aplicación

Arquitectura modular basada en el libro:
'Visión por Computador: Imágenes Digitales y Aplicaciones'
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image
import os
import sys

# Asegurar que el directorio actual está en el path
if os.path.dirname(__file__):
    sys.path.insert(0, os.path.dirname(__file__))

# Importar módulos
from core import ImageManager, HistoryManager
from gui import PhotoEditorGUI
from transforms import rotate_image, quick_rotate, scale_image, flip_image
from filters import apply_basic_filter, apply_all_adjustments
from edge_detection import (
    GradientOperator, SobelOperator, PrewittOperator, RobertsOperator,
    KirschOperator, RobinsonOperator, FreiChenOperator,
    CannyOperator, LaplacianOperator, ExtendedSobelOperator
)
from utils.normalization import normalize_to_uint8, apply_threshold, angle_to_image
from utils.validators import validate_image_array, validate_image_size
from utils.constants import MAX_IMAGE_SIZE


class PhotoEditorController:
    """Controlador principal que conecta GUI con lógica de negocio"""

    def __init__(self):
        self.image_manager = ImageManager()
        self.history_manager = HistoryManager()
        self.gui = None

        # Operadores de detección de bordes
        self.edge_operators = {
            'gradient': GradientOperator(),
            'sobel': SobelOperator(),
            'prewitt': PrewittOperator(),
            'roberts': RobertsOperator(),
            'kirsch': KirschOperator(),
            'robinson': RobinsonOperator(),
            'frei_chen': FreiChenOperator(),
            'canny': CannyOperator(),
            'laplacian': LaplacianOperator()
        }

    def set_gui(self, gui):
        """Establecer referencia a la GUI"""
        self.gui = gui

    # ========== CARGA Y GUARDADO ==========

    def load_image(self):
        """Cargar imagen desde archivo"""
        filepath = self.gui.ask_open_filename()

        if not filepath:
            return

        try:
            image = self.image_manager.load_image(filepath)
            self.history_manager.initialize_with_image(image)

            self.gui.reset_controls()
            self.update_display()
            self.update_info()
            self.gui.update_status(f"PhotoEscom - Imagen cargada: {self.image_manager.filename}")

        except Exception as e:
            self.gui.show_error("Error", f"No se pudo cargar la imagen: {str(e)}")

    def save_image(self):
        """Guardar imagen actual"""
        if not self.image_manager.has_image():
            self.gui.show_warning("Advertencia", "No hay imagen para guardar")
            return

        filepath = self.gui.ask_save_filename()

        if not filepath:
            return

        try:
            self.image_manager.save_image(filepath)
            self.gui.show_info("Éxito", "Imagen guardada correctamente")
            self.gui.update_status(f"PhotoEscom - Guardado: {os.path.basename(filepath)}")
        except Exception as e:
            self.gui.show_error("Error", f"No se pudo guardar: {str(e)}")

    # ========== HISTORIAL ==========

    def undo(self):
        """Deshacer última operación"""
        image = self.history_manager.undo()
        if image:
            self.image_manager.set_current_image(image)
            self.update_display()
            self.update_info()

    def redo(self):
        """Rehacer operación"""
        image = self.history_manager.redo()
        if image:
            self.image_manager.set_current_image(image)
            self.update_display()
            self.update_info()

    def reset_image(self):
        """Restaurar imagen original"""
        if not self.image_manager.has_image():
            return

        try:
            self.image_manager.reset_to_original()
            self.history_manager.initialize_with_image(self.image_manager.get_current_image())
            self.gui.reset_controls()
            self.update_display()
            self.image_manager.set_zoom(1.0)
        except Exception as e:
            self.gui.show_error("Error", f"No se pudo restaurar: {str(e)}")

    def add_to_history(self):
        """Añadir imagen actual al historial"""
        if self.image_manager.has_image():
            self.history_manager.add(self.image_manager.get_current_image())
            self.update_info()

    # ========== TRANSFORMACIONES ==========

    def quick_rotate(self, angle):
        """Rotación rápida de 90 grados"""
        if not self.image_manager.has_image():
            return

        try:
            current = self.image_manager.get_current_image()
            rotated = quick_rotate(current, angle)
            self.image_manager.set_current_image(rotated)
            self.add_to_history()
            self.update_display()
        except Exception as e:
            self.gui.show_error("Error", f"Error en rotación: {str(e)}")

    def preview_transform(self, transform_type):
        """Vista previa de transformación"""
        if not self.image_manager.has_image():
            return

        try:
            current = self.history_manager.get_current()
            if not current:
                return

            if transform_type == 'rotate':
                angle = self.gui.get_variable('rotate').get()
                transformed = rotate_image(current, angle)
            elif transform_type == 'scale':
                scale_x = self.gui.get_variable('scale_x').get()
                scale_y = self.gui.get_variable('scale_y').get()
                transformed = scale_image(current, scale_x, scale_y)
            else:
                return

            self.image_manager.set_current_image(transformed)
            self.update_display()
        except Exception as e:
            print(f"Error en preview: {str(e)}")

    def apply_transforms(self):
        """Aplicar transformaciones"""
        if not self.image_manager.has_image():
            return

        try:
            current = self.history_manager.get_current()
            angle = self.gui.get_variable('rotate').get()
            scale_x = self.gui.get_variable('scale_x').get()
            scale_y = self.gui.get_variable('scale_y').get()

            if angle != 0:
                current = rotate_image(current, angle)

            if scale_x != 1.0 or scale_y != 1.0:
                current = scale_image(current, scale_x, scale_y)

            self.image_manager.set_current_image(current)
            self.add_to_history()
            self.update_display()

            # Resetear valores
            self.gui.get_variable('rotate').set(0)
            self.gui.get_variable('scale_x').set(1.0)
            self.gui.get_variable('scale_y').set(1.0)

        except Exception as e:
            self.gui.show_error("Error", f"Error al aplicar transformaciones: {str(e)}")

    def apply_flip(self, direction):
        """Aplicar volteo"""
        if not self.image_manager.has_image():
            return

        try:
            current = self.image_manager.get_current_image()
            flipped = flip_image(current, direction)
            self.image_manager.set_current_image(flipped)
            self.add_to_history()
            self.update_display()
        except Exception as e:
            self.gui.show_error("Error", f"Error al voltear: {str(e)}")

    # ========== AJUSTES ==========

    def apply_adjustments(self):
        """Aplicar ajustes en tiempo real"""
        if not self.image_manager.has_image():
            return

        try:
            current = self.history_manager.get_current()

            brightness = self.gui.get_variable('brightness').get()
            contrast = self.gui.get_variable('contrast').get()
            saturation = self.gui.get_variable('saturation').get()
            sharpen = self.gui.get_variable('sharpen').get()

            adjusted = apply_all_adjustments(
                current,
                brightness,
                contrast,
                saturation,
                sharpen
            )

            self.image_manager.set_current_image(adjusted)
            self.update_display()
        except Exception as e:
            print(f"Error en ajustes: {str(e)}")

    def finalize_adjustments(self):
        """Finalizar ajustes y añadir al historial"""
        if not self.image_manager.has_image():
            return

        brightness = self.gui.get_variable('brightness').get()
        contrast = self.gui.get_variable('contrast').get()
        saturation = self.gui.get_variable('saturation').get()
        sharpen = self.gui.get_variable('sharpen').get()

        if brightness != 1.0 or contrast != 1.0 or saturation != 1.0 or sharpen != 1.0:
            self.add_to_history()
            self.gui.get_variable('brightness').set(1.0)
            self.gui.get_variable('contrast').set(1.0)
            self.gui.get_variable('saturation').set(1.0)
            self.gui.get_variable('sharpen').set(1.0)

    # ========== FILTROS ==========

    def apply_filter(self):
        """Aplicar filtro básico"""
        if not self.image_manager.has_image():
            return

        try:
            current = self.history_manager.get_current()
            filter_type = self.gui.get_variable('filter').get()

            filtered = apply_basic_filter(current, filter_type)
            self.image_manager.set_current_image(filtered)
            self.update_display()

            if filter_type != "original":
                self.add_to_history()
        except Exception as e:
            self.gui.show_error("Error", f"Error al aplicar filtro: {str(e)}")

    # ========== DETECCIÓN DE BORDES ==========

    def apply_edge_detection(self):
        """Aplicar detección de bordes"""
        if not self.image_manager.has_image():
            self.gui.show_warning("Advertencia", "No hay imagen cargada")
            return

        try:
            # Obtener imagen en escala de grises
            gray_image = self.history_manager.get_current().convert('L')
            image_array = np.array(gray_image, dtype=np.float32)

            # Validar tamaño
            is_valid, warning = validate_image_size(image_array, MAX_IMAGE_SIZE)
            if not is_valid:
                response = self.gui.ask_yes_no("Imagen Grande", warning + " ¿Desea continuar?")
                if not response:
                    return

            # Obtener parámetros
            operator_name = self.gui.get_variable('edge_operator').get()
            threshold = self.gui.get_variable('threshold').get()
            sigma = self.gui.get_variable('sigma').get()
            show_magnitude = self.gui.get_variable('show_magnitude').get()
            show_angle = self.gui.get_variable('show_angle').get()

            # Obtener operador
            operator = self._get_edge_operator(operator_name)

            # Aplicar operador
            magnitude, angle, grad_x, grad_y = operator.apply(image_array)

            # Determinar qué mostrar
            if show_magnitude:
                result = normalize_to_uint8(magnitude)
            elif show_angle and angle is not None:
                result = angle_to_image(angle)
            else:
                # Aplicar umbralización (Ecuación 6.5)
                result = apply_threshold(magnitude, threshold)

            # Convertir a imagen PIL
            edge_image = Image.fromarray(result).convert('RGB')

            self.image_manager.set_current_image(edge_image)
            self.add_to_history()
            self.update_display()
            self.gui.update_status(f"PhotoEscom - Operador {operator_name} aplicado")

        except Exception as e:
            self.gui.show_error("Error", f"Error al aplicar detección de bordes: {str(e)}")
            import traceback
            traceback.print_exc()

    def preview_edge_detection(self):
        """Vista previa rápida de detección de bordes"""
        if not self.image_manager.has_image():
            self.gui.show_warning("Advertencia", "No hay imagen cargada")
            return

        try:
            original_img = self.history_manager.get_current()

            # Reducir tamaño para preview
            if max(original_img.size) > 800:
                ratio = 800 / max(original_img.size)
                new_size = (int(original_img.size[0] * ratio),
                            int(original_img.size[1] * ratio))
                preview_img = original_img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                preview_img = original_img

            gray_image = preview_img.convert('L')
            image_array = np.array(gray_image, dtype=np.float32)

            operator_name = self.gui.get_variable('edge_operator').get()
            operator = self._get_edge_operator(operator_name)

            magnitude, angle, _, _ = operator.apply(image_array)

            threshold = self.gui.get_variable('threshold').get()
            result = apply_threshold(magnitude, threshold)

            # Redimensionar si fue reducido
            if max(original_img.size) > 800:
                result_pil = Image.fromarray(result)
                result_pil = result_pil.resize(original_img.size, Image.Resampling.NEAREST)
                result = np.array(result_pil)

            edge_image = Image.fromarray(result).convert('RGB')
            self.image_manager.set_current_image(edge_image)
            self.update_display()
            self.gui.update_status(f"PhotoEscom - Vista previa: {operator_name}")

        except Exception as e:
            self.gui.show_error("Error", f"Error en vista previa: {str(e)}")

    def _get_edge_operator(self, operator_name):
        """Obtener operador configurado"""
        if operator_name == 'roberts':
            form = self.gui.get_variable('roberts_form').get()
            operator = RobertsOperator(form=form)
        elif operator_name == 'canny':
            sigma = self.gui.get_variable('sigma').get()
            low = self.gui.get_variable('canny_low').get()
            high = self.gui.get_variable('canny_high').get()
            operator = CannyOperator(sigma=sigma, low_threshold=low, high_threshold=high)
        elif operator_name == 'laplacian':
            sigma = self.gui.get_variable('sigma').get()
            operator = LaplacianOperator(sigma=sigma)
        elif operator_name == 'sobel':
            size = self.gui.get_variable('extended_size').get()
            if size > 3:
                operator = ExtendedSobelOperator(size=size)
            else:
                operator = self.edge_operators[operator_name]
        else:
            operator = self.edge_operators.get(operator_name)

        return operator

    # ========== ZOOM ==========

    def zoom_in(self):
        """Acercar zoom"""
        if self.image_manager.has_image():
            self.image_manager.set_zoom(self.image_manager.get_zoom() * 1.2)
            self.update_display()

    def zoom_out(self):
        """Alejar zoom"""
        if self.image_manager.has_image():
            self.image_manager.set_zoom(self.image_manager.get_zoom() / 1.2)
            self.update_display()

    def zoom_fit(self):
        """Ajustar zoom a 100%"""
        if self.image_manager.has_image():
            self.image_manager.set_zoom(1.0)
            self.update_display()

    # ========== ACTUALIZACIÓN DE GUI ==========

    def update_display(self):
        """Actualizar visualización de la imagen"""
        if self.image_manager.has_image():
            self.gui.display_image(
                self.image_manager.get_current_image(),
                self.image_manager.get_zoom()
            )

    def update_info(self):
        """Actualizar información de la imagen"""
        if self.image_manager.has_image():
            info = self.image_manager.get_image_info()
            hist_info = self.history_manager.get_info()

            info_text = f"""Archivo: {info['filename']}
Dimensiones: {info['width']} x {info['height']}
Historial: {hist_info['current']}/{hist_info['total']}"""

            self.gui.update_info(info_text)
        else:
            self.gui.update_info("Sin imagen cargada")


def main():
    """Función principal"""
    root = tk.Tk()

    # Crear controlador
    controller = PhotoEditorController()

    # Crear GUI
    gui = PhotoEditorGUI(root, controller)
    controller.set_gui(gui)

    # Iniciar aplicación
    root.mainloop()


if __name__ == "__main__":
    main()