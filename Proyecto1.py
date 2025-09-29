import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import os
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class PhotoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("PhotoEscom - Editor de Fotos Profesional")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2e2e2e')
        self.root.minsize(1200, 700)

        # Variables para manejar la imagen
        self.original_image = None
        self.current_image = None
        self.display_image = None
        self.history = []
        self.history_index = -1
        self.filename = None
        self.zoom_factor = 1.0

        # Variables para controles básicos
        self.rotate_var = tk.DoubleVar(value=0)
        self.scale_x_var = tk.DoubleVar(value=1.0)
        self.scale_y_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.sharpen_var = tk.DoubleVar(value=1.0)
        self.filter_var = tk.StringVar(value="original")

        # Variables para detección de bordes
        self.edge_operator_var = tk.StringVar(value="sobel")
        self.threshold_var = tk.DoubleVar(value=50.0)
        self.sigma_var = tk.DoubleVar(value=1.0)
        self.canny_low_var = tk.DoubleVar(value=50.0)
        self.canny_high_var = tk.DoubleVar(value=150.0)

        # Configurar estilo
        self.setup_styles()

        # Crear interfaz
        self.create_widgets()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configurar colores
        self.bg_color = '#2e2e2e'
        self.frame_bg = '#3c3c3c'
        self.button_bg = '#4a4a4a'
        self.accent_color = '#007acc'
        self.text_color = '#ffffff'
        self.highlight_color = '#4a76cf'

        # Configurar estilos
        self.style.configure('TFrame', background=self.frame_bg)
        self.style.configure('TLabel', background=self.frame_bg, foreground=self.text_color)
        self.style.configure('TButton', background=self.button_bg, foreground=self.text_color,
                             borderwidth=1, focuscolor=self.accent_color)
        self.style.configure('TScale', background=self.frame_bg, troughcolor=self.accent_color)
        self.style.configure('TCheckbutton', background=self.frame_bg, foreground=self.text_color)
        self.style.configure('TRadiobutton', background=self.frame_bg, foreground=self.text_color)
        self.style.configure('TNotebook', background=self.bg_color)
        self.style.configure('TNotebook.Tab', background=self.button_bg, foreground=self.text_color,
                             padding=[10, 5])
        self.style.map('TNotebook.Tab', background=[('selected', self.accent_color)])
        self.style.map('TButton', background=[('active', self.highlight_color)])

    def create_widgets(self):
        # Barra de herramientas superior
        self.create_top_toolbar()

        # Panel principal
        main_panel = ttk.Frame(self.root)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Panel de herramientas izquierdo
        tools_notebook = ttk.Notebook(main_panel, width=350)
        tools_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tools_notebook.pack_propagate(False)

        # Pestañas
        basic_tools_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(basic_tools_frame, text="Herramientas")

        transform_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(transform_frame, text="Transformar")

        adjust_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(adjust_frame, text="Ajustes")

        filter_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(filter_frame, text="Filtros")

        edge_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(edge_frame, text="Detección Bordes")

        # Panel de visualización
        self.image_frame = ttk.Frame(main_panel)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Crear lienzo
        self.create_image_canvas()

        # Llenar pestañas
        self.create_basic_tools_panel(basic_tools_frame)
        self.create_transform_panel(transform_frame)
        self.create_adjustments_panel(adjust_frame)
        self.create_filters_panel(filter_frame)
        self.create_edge_detection_panel(edge_frame)

        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="PhotoEscom - Listo. Cargue una imagen para comenzar")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_edge_detection_panel(self, parent):
        # Canvas con scrollbar
        canvas = tk.Canvas(parent, bg=self.frame_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Operadores básicos
        basic_ops_frame = ttk.LabelFrame(scrollable_frame, text="Operadores de Primera Derivada", padding=10)
        basic_ops_frame.pack(fill=tk.X, pady=(0, 10))

        operators = [
            ("Gradiente Básico", "gradient"),
            ("Sobel", "sobel"),
            ("Prewitt", "prewitt"),
            ("Roberts", "roberts")
        ]

        for i, (text, value) in enumerate(operators):
            ttk.Radiobutton(basic_ops_frame, text=text, value=value,
                            variable=self.edge_operator_var).grid(row=i // 2, column=i % 2, sticky=tk.W, pady=2)

        # Operadores compass
        compass_frame = ttk.LabelFrame(scrollable_frame, text="Operadores Compass", padding=10)
        compass_frame.pack(fill=tk.X, pady=(0, 10))

        compass_ops = [
            ("Kirsch (8 direcciones)", "kirsch"),
            ("Robinson (8 direcciones)", "robinson")
        ]

        for i, (text, value) in enumerate(compass_ops):
            ttk.Radiobutton(compass_frame, text=text, value=value,
                            variable=self.edge_operator_var).grid(row=0, column=i, sticky=tk.W, pady=2)

        # Métodos avanzados
        advanced_frame = ttk.LabelFrame(scrollable_frame, text="Métodos Avanzados", padding=10)
        advanced_frame.pack(fill=tk.X, pady=(0, 10))

        advanced_ops = [
            ("Frei-Chen", "frei_chen"),
            ("Canny (Optimal)", "canny")
        ]

        for i, (text, value) in enumerate(advanced_ops):
            ttk.Radiobutton(advanced_frame, text=text, value=value,
                            variable=self.edge_operator_var).grid(row=0, column=i, sticky=tk.W, pady=2)

        # Parámetros
        params_frame = ttk.LabelFrame(scrollable_frame, text="Parámetros", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Umbral general
        ttk.Label(params_frame, text="Umbral:").grid(row=0, column=0, sticky=tk.W, pady=5)
        threshold_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.threshold_var)
        threshold_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        threshold_value = ttk.Label(params_frame, text="50", width=5)
        threshold_value.grid(row=0, column=2, padx=(5, 0), pady=5)

        # Sigma para Canny
        ttk.Label(params_frame, text="Sigma (Canny):").grid(row=1, column=0, sticky=tk.W, pady=5)
        sigma_scale = ttk.Scale(params_frame, from_=0.5, to=3.0, variable=self.sigma_var)
        sigma_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        sigma_value = ttk.Label(params_frame, text="1.0", width=5)
        sigma_value.grid(row=1, column=2, padx=(5, 0), pady=5)

        # Canny bajo
        ttk.Label(params_frame, text="Canny Bajo:").grid(row=2, column=0, sticky=tk.W, pady=5)
        canny_low_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.canny_low_var)
        canny_low_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        canny_low_value = ttk.Label(params_frame, text="50", width=5)
        canny_low_value.grid(row=2, column=2, padx=(5, 0), pady=5)

        # Canny alto
        ttk.Label(params_frame, text="Canny Alto:").grid(row=3, column=0, sticky=tk.W, pady=5)
        canny_high_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.canny_high_var)
        canny_high_scale.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        canny_high_value = ttk.Label(params_frame, text="150", width=5)
        canny_high_value.grid(row=3, column=2, padx=(5, 0), pady=5)

        params_frame.columnconfigure(1, weight=1)

        # Actualizar valores
        def update_param_values(*args):
            threshold_value.config(text=f"{self.threshold_var.get():.0f}")
            sigma_value.config(text=f"{self.sigma_var.get():.1f}")
            canny_low_value.config(text=f"{self.canny_low_var.get():.0f}")
            canny_high_value.config(text=f"{self.canny_high_var.get():.0f}")

        self.threshold_var.trace('w', update_param_values)
        self.sigma_var.trace('w', update_param_values)
        self.canny_low_var.trace('w', update_param_values)
        self.canny_high_var.trace('w', update_param_values)

        # Información del operador
        info_frame = ttk.LabelFrame(scrollable_frame, text="Información del Operador", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.operator_info = ttk.Label(info_frame, text="Seleccione un operador para ver información",
                                       wraplength=300, justify=tk.LEFT)
        self.operator_info.pack(anchor=tk.W)

        # Actualizar información cuando cambie el operador
        self.edge_operator_var.trace('w', self.update_operator_info)

        # Botones
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="Vista Previa",
                   command=self.preview_edge_detection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Aplicar Detección",
                   command=self.apply_edge_detection).pack(side=tk.LEFT)

        # Mouse wheel binding
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)

    def update_operator_info(self, *args):
        """Actualizar información del operador seleccionado"""
        operator = self.edge_operator_var.get()

        info_texts = {
            "gradient": "Gradiente Básico: Diferencias finitas simples. Rápido pero sensible al ruido.",
            "sobel": "Sobel: Balance óptimo entre precisión y suavizado. Excelente para uso general.",
            "prewitt": "Prewitt: Similar a Sobel con coeficientes uniformes. Menos suavizado.",
            "roberts": "Roberts: Operador diagonal simple. Bordes finos, sensible al ruido.",
            "kirsch": "Kirsch: 8 máscaras direccionales. Excelente detección multidireccional.",
            "robinson": "Robinson: Version optimizada de compass. Buena detección direccional.",
            "frei_chen": "Frei-Chen: Base vectorial completa. Separación matemática bordes/líneas.",
            "canny": "Canny: Algoritmo óptimo. Bordes finos, continuos y con mínimo ruido."
        }

        self.operator_info.config(text=info_texts.get(operator, "Información no disponible"))

    # ============= MÉTODOS DE DETECCIÓN DE BORDES CORREGIDOS =============

    def normalize_image(self, image):
        """Normalizar imagen a rango [0, 255] de forma robusta"""
        if image.dtype != np.float64:
            image = image.astype(np.float64)

        min_val = np.min(image)
        max_val = np.max(image)

        if max_val == min_val:
            return np.zeros_like(image, dtype=np.uint8)

        normalized = (image - min_val) / (max_val - min_val)
        return (normalized * 255).astype(np.uint8)

    def gradient_operator(self, image_array):
        """Gradiente básico usando diferencias finitas - CORREGIDO"""
        # Según los apuntes: diferencias finitas básicas
        # Gx = [f(x+1,y) - f(x-1,y)] / 2
        # Gy = [f(x,y+1) - f(x,y-1)] / 2

        grad_x = np.zeros_like(image_array)
        grad_y = np.zeros_like(image_array)

        # Diferencias finitas centradas
        grad_x[:, 1:-1] = (image_array[:, 2:] - image_array[:, :-2]) / 2.0
        grad_y[1:-1, :] = (image_array[2:, :] - image_array[:-2, :]) / 2.0

        # Manejar bordes
        grad_x[:, 0] = image_array[:, 1] - image_array[:, 0]
        grad_x[:, -1] = image_array[:, -1] - image_array[:, -2]
        grad_y[0, :] = image_array[1, :] - image_array[0, :]
        grad_y[-1, :] = image_array[-1, :] - image_array[-2, :]

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return self.normalize_image(magnitude)

    def sobel_operator(self, image_array):
        """Operador de Sobel - Verificado según apuntes"""
        # Máscaras exactas de los apuntes
        gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)

        gy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return self.normalize_image(magnitude)

    def prewitt_operator(self, image_array):
        """Operador de Prewitt - Verificado según apuntes"""
        # Máscaras exactas de los apuntes
        gx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float64)

        gy = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return self.normalize_image(magnitude)

    def roberts_operator(self, image_array):
        """Operador de Roberts - Implementación corregida"""
        # Máscaras diagonales según los apuntes
        gx = np.array([[1, 0],
                       [0, -1]], dtype=np.float64)

        gy = np.array([[0, 1],
                       [-1, 0]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        # Forma 1 de los apuntes: R = sqrt(D1² + D2²)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return self.normalize_image(magnitude)

    def kirsch_operator(self, image_array):
        """Operador de Kirsch - Verificado según apuntes"""
        # Las 8 máscaras exactas de los apuntes
        masks = [
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # 0°
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # 45°
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # 90°
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # 135°
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # 180°
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # 225°
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # 270°
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])  # 315°
        ]

        responses = []
        for mask in masks:
            response = ndimage.convolve(image_array.astype(np.float64),
                                        mask.astype(np.float64))
            responses.append(response)

        # Máximo en cada posición según los apuntes
        max_response = np.maximum.reduce(responses)
        return self.normalize_image(max_response)

    def robinson_operator(self, image_array):
        """Operador de Robinson - CORREGIDO las 8 direcciones"""
        # Las 8 máscaras correctas (cada 45°) basadas en rotación de máscara base
        masks = [
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # 0°
            np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),  # 45°
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # 90°
            np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),  # 135°
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),  # 180°
            np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),  # 225°
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),  # 270°
            np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])  # 315°
        ]

        responses = []
        for mask in masks:
            response = ndimage.convolve(image_array.astype(np.float64),
                                        mask.astype(np.float64))
            responses.append(response)

        max_response = np.maximum.reduce(responses)
        return self.normalize_image(max_response)

    def frei_chen_operator(self, image_array):
        """Operador de Frei-Chen - Verificado según apuntes"""
        sqrt2 = np.sqrt(2)

        # Las 9 máscaras exactas de los apuntes
        masks = [
            # Bordes (f1-f4)
            np.array([[1, sqrt2, 1], [0, 0, 0], [-1, -sqrt2, -1]]) / (2 * sqrt2),
            np.array([[1, 0, -1], [sqrt2, 0, -sqrt2], [1, 0, -1]]) / (2 * sqrt2),
            np.array([[0, -1, sqrt2], [1, 0, -1], [-sqrt2, 1, 0]]) / (2 * sqrt2),
            np.array([[sqrt2, -1, 0], [-1, 0, 1], [0, 1, -sqrt2]]) / (2 * sqrt2),
            # Líneas (f5-f8)
            np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]]) / 2,
            np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / 2,
            np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 6,
            np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]]) / 6,
            # Promedio (f9)
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 3
        ]

        # Solo usar las primeras 4 máscaras para bordes
        edge_responses = []
        for i in range(4):
            response = ndimage.convolve(image_array.astype(np.float64),
                                        masks[i].astype(np.float64))
            edge_responses.append(response ** 2)

        # Magnitud de bordes según los apuntes
        edge_magnitude = np.sqrt(np.sum(edge_responses, axis=0))
        return self.normalize_image(edge_magnitude)

    def canny_operator(self, image_array):
        """Algoritmo de Canny - Optimizado usando OpenCV cuando es posible"""
        try:
            # Usar OpenCV si está disponible (mucho más eficiente)
            img_uint8 = image_array.astype(np.uint8)

            # Aplicar filtro gaussiano
            sigma = self.sigma_var.get()
            ksize = int(2 * np.ceil(2 * sigma) + 1)  # Tamaño apropiado del kernel
            if ksize % 2 == 0:
                ksize += 1

            blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)

            # Aplicar Canny
            low_threshold = int(self.canny_low_var.get())
            high_threshold = int(self.canny_high_var.get())

            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            return edges

        except Exception as e:
            # Fallback a implementación manual si OpenCV falla
            return self.canny_manual(image_array)

    def canny_manual(self, image_array):
        """Implementación manual de Canny (fallback)"""
        # 1. Suavizado gaussiano
        sigma = self.sigma_var.get()
        smoothed = gaussian_filter(image_array.astype(np.float64), sigma=sigma)

        # 2. Gradiente usando Sobel
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(smoothed, gx)
        grad_y = ndimage.convolve(smoothed, gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        # 3. Supresión no máxima (versión optimizada)
        suppressed = self.non_maximum_suppression_optimized(magnitude, angle)

        # 4. Histéresis
        low_threshold = self.canny_low_var.get()
        high_threshold = self.canny_high_var.get()

        edges = self.hysteresis_threshold_optimized(suppressed, low_threshold, high_threshold)

        return (edges * 255).astype(np.uint8)

    def non_maximum_suppression_optimized(self, magnitude, angle):
        """Supresión no máxima optimizada"""
        rows, cols = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # Convertir ángulos a grados y normalizar
        angle_deg = (np.rad2deg(angle) + 180) % 180

        # Vectorizar la clasificación de direcciones
        dir_mask = np.zeros_like(angle_deg, dtype=int)
        dir_mask[(angle_deg >= 157.5) | (angle_deg < 22.5)] = 0  # Horizontal
        dir_mask[(angle_deg >= 22.5) & (angle_deg < 67.5)] = 1  # Diagonal /
        dir_mask[(angle_deg >= 67.5) & (angle_deg < 112.5)] = 2  # Vertical
        dir_mask[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 3  # Diagonal \

        # Aplicar supresión para cada dirección
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                direction = dir_mask[i, j]
                current = magnitude[i, j]

                if direction == 0:  # Horizontal
                    neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
                elif direction == 1:  # Diagonal /
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
                elif direction == 2:  # Vertical
                    neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
                else:  # Diagonal \
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

                if current >= max(neighbors):
                    suppressed[i, j] = current

        return suppressed

    def hysteresis_threshold_optimized(self, image, low_threshold, high_threshold):
        """Histéresis optimizada usando conectividad"""
        strong_edges = image > high_threshold
        weak_edges = (image >= low_threshold) & (image <= high_threshold)

        # Usar ndimage para encontrar componentes conectados
        from scipy.ndimage import label, binary_dilation

        # Dilatar bordes fuertes para conectar con débiles cercanos
        strong_dilated = binary_dilation(strong_edges, structure=np.ones((3, 3)))

        # Mantener solo bordes débiles conectados a fuertes
        connected_weak = weak_edges & strong_dilated

        # Resultado final
        edges = strong_edges | connected_weak

        return edges.astype(np.float32)

    def apply_edge_detection(self):
        """Aplicar detección de bordes con validaciones"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen cargada")
            return

        try:
            # Obtener imagen en escala de grises
            gray_image = self.history[self.history_index].convert('L')
            image_array = np.array(gray_image, dtype=np.float32)

            # Validar tamaño de imagen
            if image_array.size > 10000000:  # ~3000x3000 pixels
                response = messagebox.askyesno("Imagen Grande",
                                               "La imagen es muy grande. ¿Desea continuar? (Puede tardar varios segundos)")
                if not response:
                    return

            operator = self.edge_operator_var.get()

            # Aplicar operador seleccionado
            if operator == "sobel":
                result = self.sobel_operator(image_array)
            elif operator == "prewitt":
                result = self.prewitt_operator(image_array)
            elif operator == "roberts":
                result = self.roberts_operator(image_array)
            elif operator == "gradient":
                result = self.gradient_operator(image_array)
            elif operator == "kirsch":
                result = self.kirsch_operator(image_array)
            elif operator == "robinson":
                result = self.robinson_operator(image_array)
            elif operator == "frei_chen":
                result = self.frei_chen_operator(image_array)
            elif operator == "canny":
                result = self.canny_operator(image_array)
            else:
                messagebox.showerror("Error", f"Operador no reconocido: {operator}")
                return

            # Aplicar umbralización si no es Canny
            if operator != "canny":
                threshold = self.threshold_var.get()
                result = np.where(result > threshold, 255, 0).astype(np.uint8)

            # Convertir a imagen PIL
            edge_image = Image.fromarray(result).convert('RGB')

            self.current_image = edge_image
            self.add_to_history()
            self.display_image_on_canvas()
            self.status_bar.config(text=f"PhotoEscom - Operador {operator} aplicado")

        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar detección de bordes: {str(e)}")
            print(f"Debug - Error: {e}")  # Para debugging

    def preview_edge_detection(self):
        """Vista previa optimizada"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen cargada")
            return

        try:
            # Usar imagen reducida para preview rápido
            original_img = self.history[self.history_index]

            # Reducir tamaño si es muy grande
            if max(original_img.size) > 800:
                ratio = 800 / max(original_img.size)
                new_size = (int(original_img.size[0] * ratio),
                            int(original_img.size[1] * ratio))
                preview_img = original_img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                preview_img = original_img

            gray_image = preview_img.convert('L')
            image_array = np.array(gray_image, dtype=np.float32)

            operator = self.edge_operator_var.get()

            # Aplicar operador
            if operator == "sobel":
                result = self.sobel_operator(image_array)
            elif operator == "prewitt":
                result = self.prewitt_operator(image_array)
            elif operator == "roberts":
                result = self.roberts_operator(image_array)
            elif operator == "gradient":
                result = self.gradient_operator(image_array)
            elif operator == "kirsch":
                result = self.kirsch_operator(image_array)
            elif operator == "robinson":
                result = self.robinson_operator(image_array)
            elif operator == "frei_chen":
                result = self.frei_chen_operator(image_array)
            elif operator == "canny":
                result = self.canny_operator(image_array)
            else:
                return

            if operator != "canny":
                threshold = self.threshold_var.get()
                result = np.where(result > threshold, 255, 0).astype(np.uint8)

            # Redimensionar resultado al tamaño original si fue reducido
            if max(original_img.size) > 800:
                result_pil = Image.fromarray(result)
                result_pil = result_pil.resize(original_img.size, Image.Resampling.NEAREST)
                result = np.array(result_pil)

            edge_image = Image.fromarray(result).convert('RGB')
            self.current_image = edge_image
            self.display_image_on_canvas()
            self.status_bar.config(text=f"PhotoEscom - Vista previa: {operator}")

        except Exception as e:
            messagebox.showerror("Error", f"Error en vista previa: {str(e)}")

    # ============= MÉTODOS ORIGINALES DEL EDITOR =============

    def create_top_toolbar(self):
        toolbar = ttk.Frame(self.root, height=50)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 5))
        toolbar.pack_propagate(False)

        title_label = ttk.Label(toolbar, text="PhotoEscom", font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=(10, 20))

        btn_load = ttk.Button(toolbar, text="📁 Cargar Imagen", command=self.load_image, width=15)
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_save = ttk.Button(toolbar, text="💾 Guardar", command=self.save_image, width=12)
        btn_save.pack(side=tk.LEFT, padx=5)

        separator = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        btn_undo = ttk.Button(toolbar, text="↶ Deshacer", command=self.undo, width=10)
        btn_undo.pack(side=tk.LEFT, padx=5)

        btn_redo = ttk.Button(toolbar, text="↷ Rehacer", command=self.redo, width=10)
        btn_redo.pack(side=tk.LEFT, padx=5)

        btn_reset = ttk.Button(toolbar, text="🔄 Restaurar", command=self.reset_image, width=12)
        btn_reset.pack(side=tk.LEFT, padx=5)

        separator2 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator2.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        zoom_frame = ttk.Frame(toolbar)
        zoom_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        btn_zoom_out = ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3)
        btn_zoom_out.pack(side=tk.LEFT, padx=2)
        btn_zoom_in = ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3)
        btn_zoom_in.pack(side=tk.LEFT, padx=2)
        btn_zoom_fit = ttk.Button(zoom_frame, text="🔍 Ajustar", command=self.zoom_fit, width=8)
        btn_zoom_fit.pack(side=tk.LEFT, padx=2)

    def create_image_canvas(self):
        canvas_container = ttk.Frame(self.image_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)

        self.canvas = tk.Canvas(canvas_container, bg='#1e1e1e', highlightthickness=0,
                                yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)

        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        v_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        h_scrollbar.grid(row=1, column=0, sticky=tk.EW)

        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.image_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")

        self.image_container.bind("<Configure>", self.on_container_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def create_basic_tools_panel(self, parent):
        info_frame = ttk.LabelFrame(parent, text="Información de la imagen", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_label = ttk.Label(info_frame, text="Sin imagen cargada", justify=tk.LEFT)
        self.info_label.pack(anchor=tk.W)

        tools_frame = ttk.LabelFrame(parent, text="Herramientas rápidas", padding=10)
        tools_frame.pack(fill=tk.X, pady=(0, 10))

        flip_frame = ttk.Frame(tools_frame)
        flip_frame.pack(fill=tk.X, pady=5)
        ttk.Button(flip_frame, text="↔ Voltear Horizontal", command=lambda: self.apply_flip('horizontal')).pack(
            side=tk.LEFT, expand=True, padx=2)
        ttk.Button(flip_frame, text="↕ Voltear Vertical", command=lambda: self.apply_flip('vertical')).pack(
            side=tk.LEFT, expand=True, padx=2)

        rotate_frame = ttk.Frame(tools_frame)
        rotate_frame.pack(fill=tk.X, pady=5)
        ttk.Button(rotate_frame, text="↺ 90°", command=lambda: self.quick_rotate(-90)).pack(side=tk.LEFT, expand=True,
                                                                                            padx=2)
        ttk.Button(rotate_frame, text="↻ 90°", command=lambda: self.quick_rotate(90)).pack(side=tk.LEFT, expand=True,
                                                                                           padx=2)

    def create_transform_panel(self, parent):
        ttk.Label(parent, text="Rotación:").grid(row=0, column=0, sticky=tk.W, pady=5)
        rotate_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.rotate_var,
                                 command=lambda e: self.preview_transform('rotate'))
        rotate_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        rotate_entry = ttk.Entry(parent, textvariable=self.rotate_var, width=5)
        rotate_entry.grid(row=0, column=2, padx=(5, 0), pady=5)

        ttk.Label(parent, text="Escala X:").grid(row=1, column=0, sticky=tk.W, pady=5)
        scale_x_scale = ttk.Scale(parent, from_=0.1, to=5.0, variable=self.scale_x_var,
                                  command=lambda e: self.preview_transform('scale'))
        scale_x_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        scale_x_entry = ttk.Entry(parent, textvariable=self.scale_x_var, width=5)
        scale_x_entry.grid(row=1, column=2, padx=(5, 0), pady=5)

        ttk.Label(parent, text="Escala Y:").grid(row=2, column=0, sticky=tk.W, pady=5)
        scale_y_scale = ttk.Scale(parent, from_=0.1, to=5.0, variable=self.scale_y_var,
                                  command=lambda e: self.preview_transform('scale'))
        scale_y_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        scale_y_entry = ttk.Entry(parent, textvariable=self.scale_y_var, width=5)
        scale_y_entry.grid(row=2, column=2, padx=(5, 0), pady=5)

        ttk.Button(parent, text="Aplicar Transformaciones", command=self.apply_transforms).grid(row=3, column=0,
                                                                                                columnspan=3, pady=10)

        parent.columnconfigure(1, weight=1)

    def create_adjustments_panel(self, parent):
        ttk.Label(parent, text="Brillo:").grid(row=0, column=0, sticky=tk.W, pady=5)
        brightness_scale = ttk.Scale(parent, from_=0.1, to=3.0, variable=self.brightness_var,
                                     command=lambda e: self.apply_adjustments())
        brightness_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        brightness_value = ttk.Label(parent, text="1.0", width=5)
        brightness_value.grid(row=0, column=2, padx=(5, 0), pady=5)

        ttk.Label(parent, text="Contraste:").grid(row=1, column=0, sticky=tk.W, pady=5)
        contrast_scale = ttk.Scale(parent, from_=0.1, to=3.0, variable=self.contrast_var,
                                   command=lambda e: self.apply_adjustments())
        contrast_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        contrast_value = ttk.Label(parent, text="1.0", width=5)
        contrast_value.grid(row=1, column=2, padx=(5, 0), pady=5)

        ttk.Label(parent, text="Saturación:").grid(row=2, column=0, sticky=tk.W, pady=5)
        saturation_scale = ttk.Scale(parent, from_=0.0, to=3.0, variable=self.saturation_var,
                                     command=lambda e: self.apply_adjustments())
        saturation_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        saturation_value = ttk.Label(parent, text="1.0", width=5)
        saturation_value.grid(row=2, column=2, padx=(5, 0), pady=5)

        ttk.Label(parent, text="Nitidez:").grid(row=3, column=0, sticky=tk.W, pady=5)
        sharpen_scale = ttk.Scale(parent, from_=1.0, to=5.0, variable=self.sharpen_var,
                                  command=lambda e: self.apply_adjustments())
        sharpen_scale.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        sharpen_value = ttk.Label(parent, text="1.0", width=5)
        sharpen_value.grid(row=3, column=2, padx=(5, 0), pady=5)

        ttk.Button(parent, text="Aplicar Ajustes", command=self.finalize_adjustments).grid(row=4, column=0,
                                                                                           columnspan=3, pady=10)

        parent.columnconfigure(1, weight=1)

        def update_values(*args):
            brightness_value.config(text=f"{self.brightness_var.get():.1f}")
            contrast_value.config(text=f"{self.contrast_var.get():.1f}")
            saturation_value.config(text=f"{self.saturation_var.get():.1f}")
            sharpen_value.config(text=f"{self.sharpen_var.get():.1f}")

        self.brightness_var.trace('w', update_values)
        self.contrast_var.trace('w', update_values)
        self.saturation_var.trace('w', update_values)
        self.sharpen_var.trace('w', update_values)

    def create_filters_panel(self, parent):
        filters = [
            ("Original", "original"),
            ("Escala de Grises", "grayscale"),
            ("Sepia", "sepia"),
            ("Invertir", "invert"),
            ("Desenfoque", "blur"),
            ("Detalle", "detail"),
            ("Bordes", "edges"),
            ("Realce", "enhance")
        ]

        self.filter_var = tk.StringVar(value="original")

        for i, (text, mode) in enumerate(filters):
            btn = ttk.Radiobutton(parent, text=text, value=mode,
                                  variable=self.filter_var, command=self.apply_filter)
            btn.grid(row=i, column=0, sticky=tk.W, pady=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )

        if file_path:
            try:
                self.original_image = Image.open(file_path).convert('RGB')
                self.current_image = self.original_image.copy()
                self.filename = os.path.basename(file_path)

                self.history = [self.current_image.copy()]
                self.history_index = 0
                self.zoom_factor = 1.0

                self.reset_controls()
                self.display_image_on_canvas()
                self.update_info()

                self.status_bar.config(text=f"PhotoEscom - Imagen cargada: {self.filename}")

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")

    def reset_controls(self):
        self.rotate_var.set(0)
        self.scale_x_var.set(1.0)
        self.scale_y_var.set(1.0)
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.sharpen_var.set(1.0)
        self.filter_var.set("original")
        self.edge_operator_var.set("sobel")
        self.threshold_var.set(50.0)
        self.sigma_var.set(1.0)
        self.canny_low_var.set(50.0)
        self.canny_high_var.set(150.0)

    def update_info(self):
        if self.current_image:
            width, height = self.current_image.size
            file_size = os.path.getsize(self.filename) if self.filename and os.path.exists(self.filename) else "N/A"

            if file_size != "N/A":
                if file_size < 1024:
                    file_size_str = f"{file_size} bytes"
                elif file_size < 1024 * 1024:
                    file_size_str = f"{file_size / 1024:.1f} KB"
                else:
                    file_size_str = f"{file_size / (1024 * 1024):.1f} MB"
            else:
                file_size_str = "N/A"

            info_text = f"""Archivo: {self.filename or 'N/A'}
Dimensiones: {width} x {height}
Tamaño: {file_size_str}
Historial: {self.history_index + 1}/{len(self.history)}"""

            self.info_label.config(text=info_text)
        else:
            self.info_label.config(text="Sin imagen cargada")

    def display_image_on_canvas(self):
        if self.current_image:
            width, height = self.current_image.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)

            img = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.display_image = ImageTk.PhotoImage(img)

            if hasattr(self, 'image_label'):
                self.image_label.config(image=self.display_image)
            else:
                self.image_label = ttk.Label(self.image_container, image=self.display_image)
                self.image_label.pack()

            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def save_image(self):
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos los archivos", "*.*")]
        )

        if file_path:
            try:
                self.current_image.save(file_path)
                messagebox.showinfo("Éxito", "Imagen guardada correctamente")
                self.status_bar.config(text=f"PhotoEscom - Imagen guardada: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen: {str(e)}")

    def add_to_history(self):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]

        self.history.append(self.current_image.copy())
        self.history_index = len(self.history) - 1

        if len(self.history) > 20:
            self.history.pop(0)
            self.history_index -= 1

        self.update_info()

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.update_info()
            self.status_bar.config(text="PhotoEscom - Deshacer: paso anterior")

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.update_info()
            self.status_bar.config(text="PhotoEscom - Rehacer: paso siguiente")

    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.add_to_history()
            self.reset_controls()
            self.display_image_on_canvas()
            self.zoom_factor = 1.0
            self.status_bar.config(text="PhotoEscom - Imagen restaurada a su estado original")

    def quick_rotate(self, angle):
        if not self.current_image:
            return

        img = self.history[self.history_index].copy()
        img = img.rotate(angle, expand=True)

        self.current_image = img
        self.add_to_history()
        self.display_image_on_canvas()
        self.status_bar.config(text=f"PhotoEscom - Rotación aplicada: {angle}°")

    def preview_transform(self, transform_type):
        if not self.current_image:
            return

        img = self.history[self.history_index].copy()

        if transform_type == 'rotate':
            angle = self.rotate_var.get()
            img = img.rotate(angle, expand=True)

        elif transform_type == 'scale':
            scale_x = self.scale_x_var.get()
            scale_y = self.scale_y_var.get()
            width, height = img.size
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.current_image = img
        self.display_image_on_canvas()

    def apply_transforms(self):
        if not self.current_image:
            return

        img = self.history[self.history_index].copy()

        angle = self.rotate_var.get()
        if angle != 0:
            img = img.rotate(angle, expand=True)

        scale_x = self.scale_x_var.get()
        scale_y = self.scale_y_var.get()
        if scale_x != 1.0 or scale_y != 1.0:
            width, height = img.size
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.current_image = img
        self.add_to_history()
        self.display_image_on_canvas()
        self.status_bar.config(text="PhotoEscom - Transformaciones aplicadas")

        self.rotate_var.set(0)
        self.scale_x_var.set(1.0)
        self.scale_y_var.set(1.0)

    def apply_flip(self, direction):
        if not self.current_image:
            return

        img = self.history[self.history_index].copy()

        if direction == 'horizontal':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        self.current_image = img
        self.add_to_history()
        self.display_image_on_canvas()
        self.status_bar.config(
            text=f"PhotoEscom - Imagen volteada {'horizontalmente' if direction == 'horizontal' else 'verticalmente'}")

    def apply_adjustments(self):
        if not self.current_image:
            return

        img = self.history[self.history_index].copy()

        brightness = self.brightness_var.get()
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        contrast = self.contrast_var.get()
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)

        saturation = self.saturation_var.get()
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)

        sharpen = self.sharpen_var.get()
        if sharpen != 1.0:
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        self.current_image = img
        self.display_image_on_canvas()

    def finalize_adjustments(self):
        if not self.current_image:
            return

        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        saturation = self.saturation_var.get()
        sharpen = self.sharpen_var.get()

        if brightness != 1.0 or contrast != 1.0 or saturation != 1.0 or sharpen != 1.0:
            self.add_to_history()
            self.status_bar.config(text="PhotoEscom - Ajustes aplicados permanentemente")

            self.brightness_var.set(1.0)
            self.contrast_var.set(1.0)
            self.saturation_var.set(1.0)
            self.sharpen_var.set(1.0)

    def apply_filter(self):
        if not self.current_image:
            return

        img = self.history[self.history_index].copy()
        filter_type = self.filter_var.get()

        if filter_type == "grayscale":
            img = img.convert("L").convert("RGB")
        elif filter_type == "sepia":
            img_array = np.array(img)
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
            sepia_img = np.dot(img_array, sepia_filter.T)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            img = Image.fromarray(sepia_img)
        elif filter_type == "invert":
            img_array = np.array(img)
            img_array = 255 - img_array
            img = Image.fromarray(img_array)
        elif filter_type == "blur":
            img = img.filter(ImageFilter.BLUR)
        elif filter_type == "detail":
            img = img.filter(ImageFilter.DETAIL)
        elif filter_type == "edges":
            img = img.filter(ImageFilter.FIND_EDGES)
        elif filter_type == "enhance":
            img = img.filter(ImageFilter.EDGE_ENHANCE)

        self.current_image = img
        self.display_image_on_canvas()

        if filter_type != "original":
            self.add_to_history()
            self.status_bar.config(text=f"PhotoEscom - Filtro aplicado: {filter_type}")
        else:
            self.status_bar.config(text="PhotoEscom - Filtro original restaurado")

    def zoom_in(self):
        if self.current_image:
            self.zoom_factor *= 1.2
            self.display_image_on_canvas()
            self.status_bar.config(text=f"PhotoEscom - Zoom: {int(self.zoom_factor * 100)}%")

    def zoom_out(self):
        if self.current_image:
            self.zoom_factor /= 1.2
            self.display_image_on_canvas()
            self.status_bar.config(text=f"PhotoEscom - Zoom: {int(self.zoom_factor * 100)}%")

    def zoom_fit(self):
        if self.current_image:
            self.zoom_factor = 1.0
            self.display_image_on_canvas()
            self.status_bar.config(text="PhotoEscom - Zoom ajustado a la imagen")

    def on_container_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width, height=event.height)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_resize(self, event):
        if event.widget == self.root:
            self.display_image_on_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoEditor(root)
    root.bind("<Configure>", app.on_resize)
    root.mainloop()