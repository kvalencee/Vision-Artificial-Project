import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import os


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

        # Variables para controles
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
        self.threshold_var = tk.DoubleVar(value=100)
        self.canny_threshold1_var = tk.DoubleVar(value=50)
        self.canny_threshold2_var = tk.DoubleVar(value=150)
        self.kernel_size_var = tk.IntVar(value=3)

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

        # Panel de herramientas izquierdo (ahora con pestañas)
        tools_notebook = ttk.Notebook(main_panel, width=300)
        tools_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tools_notebook.pack_propagate(False)

        # Pestaña de herramientas básicas
        basic_tools_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(basic_tools_frame, text="Herramientas")

        # Pestaña de transformaciones
        transform_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(transform_frame, text="Transformar")

        # Pestaña de ajustes
        adjust_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(adjust_frame, text="Ajustes")

        # Pestaña de filtros
        filter_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(filter_frame, text="Filtros")

        # Pestaña de detección de bordes (NUEVA)
        edge_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(edge_frame, text="Bordes")

        # Panel de visualización de imagen
        self.image_frame = ttk.Frame(main_panel)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Crear lienzo para la imagen con scrollbars
        self.create_image_canvas()

        # Llenar las pestañas con contenido
        self.create_basic_tools_panel(basic_tools_frame)
        self.create_transform_panel(transform_frame)
        self.create_adjustments_panel(adjust_frame)
        self.create_filters_panel(filter_frame)
        self.create_edge_detection_panel(edge_frame)  # NUEVO

        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="PhotoEscom - Listo. Cargue una imagen para comenzar")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_top_toolbar(self):
        toolbar = ttk.Frame(self.root, height=50)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 5))
        toolbar.pack_propagate(False)

        # Título de la aplicación
        title_label = ttk.Label(toolbar, text="PhotoEscom", font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=(10, 20))

        # Botones de control de archivo
        btn_load = ttk.Button(toolbar, text="📁 Cargar Imagen", command=self.load_image, width=15)
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_save = ttk.Button(toolbar, text="💾 Guardar", command=self.save_image, width=12)
        btn_save.pack(side=tk.LEFT, padx=5)

        # Separador
        separator = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Botones de navegación de historial
        btn_undo = ttk.Button(toolbar, text="↶ Deshacer", command=self.undo, width=10)
        btn_undo.pack(side=tk.LEFT, padx=5)

        btn_redo = ttk.Button(toolbar, text="↷ Rehacer", command=self.redo, width=10)
        btn_redo.pack(side=tk.LEFT, padx=5)

        btn_reset = ttk.Button(toolbar, text="🔄 Restaurar", command=self.reset_image, width=12)
        btn_reset.pack(side=tk.LEFT, padx=5)

        # Separador
        separator2 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator2.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Controles de zoom
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
        # Frame para el canvas con scrollbars
        canvas_container = ttk.Frame(self.image_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)

        # Canvas para la imagen
        self.canvas = tk.Canvas(canvas_container, bg='#1e1e1e', highlightthickness=0,
                                yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Configurar scrollbars
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        v_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        h_scrollbar.grid(row=1, column=0, sticky=tk.EW)

        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        # Frame interno para la imagen
        self.image_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")

        # Configurar eventos
        self.image_container.bind("<Configure>", self.on_container_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def create_basic_tools_panel(self, parent):
        # Información de la imagen
        info_frame = ttk.LabelFrame(parent, text="Información de la imagen", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_label = ttk.Label(info_frame, text="Sin imagen cargada", justify=tk.LEFT)
        self.info_label.pack(anchor=tk.W)

        # Herramientas rápidas
        tools_frame = ttk.LabelFrame(parent, text="Herramientas rápidas", padding=10)
        tools_frame.pack(fill=tk.X, pady=(0, 10))

        # Botones de volteo
        flip_frame = ttk.Frame(tools_frame)
        flip_frame.pack(fill=tk.X, pady=5)
        ttk.Button(flip_frame, text="↔ Voltear Horizontal", command=lambda: self.apply_flip('horizontal')).pack(
            side=tk.LEFT, expand=True, padx=2)
        ttk.Button(flip_frame, text="↕ Voltear Vertical", command=lambda: self.apply_flip('vertical')).pack(
            side=tk.LEFT, expand=True, padx=2)

        # Botones de rotación rápida
        rotate_frame = ttk.Frame(tools_frame)
        rotate_frame.pack(fill=tk.X, pady=5)
        ttk.Button(rotate_frame, text="↺ 90°", command=lambda: self.quick_rotate(-90)).pack(side=tk.LEFT, expand=True,
                                                                                            padx=2)
        ttk.Button(rotate_frame, text="↻ 90°", command=lambda: self.quick_rotate(90)).pack(side=tk.LEFT, expand=True,
                                                                                           padx=2)

    def create_transform_panel(self, parent):
        # Rotación
        ttk.Label(parent, text="Rotación:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.rotate_var = tk.DoubleVar(value=0)
        rotate_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.rotate_var,
                                 command=lambda e: self.preview_transform('rotate'))
        rotate_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        rotate_entry = ttk.Entry(parent, textvariable=self.rotate_var, width=5)
        rotate_entry.grid(row=0, column=2, padx=(5, 0), pady=5)

        # Escala
        ttk.Label(parent, text="Escala X:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.scale_x_var = tk.DoubleVar(value=1.0)
        scale_x_scale = ttk.Scale(parent, from_=0.1, to=5.0, variable=self.scale_x_var,
                                  command=lambda e: self.preview_transform('scale'))
        scale_x_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        scale_x_entry = ttk.Entry(parent, textvariable=self.scale_x_var, width=5)
        scale_x_entry.grid(row=1, column=2, padx=(5, 0), pady=5)

        ttk.Label(parent, text="Escala Y:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.scale_y_var = tk.DoubleVar(value=1.0)
        scale_y_scale = ttk.Scale(parent, from_=0.1, to=5.0, variable=self.scale_y_var,
                                  command=lambda e: self.preview_transform('scale'))
        scale_y_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        scale_y_entry = ttk.Entry(parent, textvariable=self.scale_y_var, width=5)
        scale_y_entry.grid(row=2, column=2, padx=(5, 0), pady=5)

        # Aplicar transformaciones
        ttk.Button(parent, text="Aplicar Transformaciones", command=self.apply_transforms).grid(row=3, column=0,
                                                                                                columnspan=3, pady=10)

        parent.columnconfigure(1, weight=1)

    def create_adjustments_panel(self, parent):
        # Brillo
        ttk.Label(parent, text="Brillo:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(parent, from_=0.1, to=3.0, variable=self.brightness_var,
                                     command=lambda e: self.apply_adjustments())
        brightness_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        brightness_value = ttk.Label(parent, text="1.0", width=5)
        brightness_value.grid(row=0, column=2, padx=(5, 0), pady=5)

        # Contraste
        ttk.Label(parent, text="Contraste:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(parent, from_=0.1, to=3.0, variable=self.contrast_var,
                                   command=lambda e: self.apply_adjustments())
        contrast_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        contrast_value = ttk.Label(parent, text="1.0", width=5)
        contrast_value.grid(row=1, column=2, padx=(5, 0), pady=5)

        # Saturación
        ttk.Label(parent, text="Saturación:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.saturation_var = tk.DoubleVar(value=1.0)
        saturation_scale = ttk.Scale(parent, from_=0.0, to=3.0, variable=self.saturation_var,
                                     command=lambda e: self.apply_adjustments())
        saturation_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        saturation_value = ttk.Label(parent, text="1.0", width=5)
        saturation_value.grid(row=2, column=2, padx=(5, 0), pady=5)

        # Nitidez
        ttk.Label(parent, text="Nitidez:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.sharpen_var = tk.DoubleVar(value=1.0)
        sharpen_scale = ttk.Scale(parent, from_=1.0, to=5.0, variable=self.sharpen_var,
                                  command=lambda e: self.apply_adjustments())
        sharpen_scale.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        sharpen_value = ttk.Label(parent, text="1.0", width=5)
        sharpen_value.grid(row=3, column=2, padx=(5, 0), pady=5)

        # Botón para aplicar ajustes permanentemente
        ttk.Button(parent, text="Aplicar Ajustes", command=self.finalize_adjustments).grid(row=4, column=0,
                                                                                           columnspan=3, pady=10)

        parent.columnconfigure(1, weight=1)

        # Actualizar etiquetas de valores
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

    def create_edge_detection_panel(self, parent):
        """NUEVO: Panel para detección de bordes"""

        # Operadores disponibles
        operators_frame = ttk.LabelFrame(parent, text="Operadores de Bordes", padding=10)
        operators_frame.pack(fill=tk.X, pady=(0, 10))

        operators = [
            ("Sobel", "sobel"),
            ("Prewitt", "prewitt"),
            ("Roberts", "roberts"),
            ("Kirsch", "kirsch"),
            ("Robinson", "robinson"),
            ("Frei-Chen", "frei_chen"),
            ("Canny", "canny"),
            ("Gradiente", "gradient")
        ]

        self.edge_operator_var = tk.StringVar(value="sobel")

        for i, (text, mode) in enumerate(operators):
            btn = ttk.Radiobutton(operators_frame, text=text, value=mode,
                                  variable=self.edge_operator_var, command=self.apply_edge_detection)
            btn.grid(row=i // 2, column=i % 2, sticky=tk.W, pady=2, padx=5)

        # Parámetros comunes
        params_frame = ttk.LabelFrame(parent, text="Parámetros", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Umbral general
        ttk.Label(params_frame, text="Umbral:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.DoubleVar(value=100)
        threshold_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.threshold_var,
                                    command=lambda e: self.apply_edge_detection())
        threshold_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        threshold_value = ttk.Label(params_frame, text="100", width=5)
        threshold_value.grid(row=0, column=2, padx=(5, 0), pady=5)

        # Tamaño del kernel
        ttk.Label(params_frame, text="Tamaño Kernel:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.kernel_size_var = tk.IntVar(value=3)
        kernel_combo = ttk.Combobox(params_frame, textvariable=self.kernel_size_var,
                                    values=[1, 3, 5, 7], state="readonly", width=5)
        kernel_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(5, 0))

        # Parámetros específicos de Canny
        canny_frame = ttk.LabelFrame(parent, text="Parámetros Canny", padding=10)
        canny_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(canny_frame, text="Umbral 1:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.canny_threshold1_var = tk.DoubleVar(value=50)
        canny1_scale = ttk.Scale(canny_frame, from_=0, to=255, variable=self.canny_threshold1_var,
                                 command=lambda e: self.apply_edge_detection())
        canny1_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        canny1_value = ttk.Label(canny_frame, text="50", width=5)
        canny1_value.grid(row=0, column=2, padx=(5, 0), pady=5)

        ttk.Label(canny_frame, text="Umbral 2:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.canny_threshold2_var = tk.DoubleVar(value=150)
        canny2_scale = ttk.Scale(canny_frame, from_=0, to=255, variable=self.canny_threshold2_var,
                                 command=lambda e: self.apply_edge_detection())
        canny2_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        canny2_value = ttk.Label(canny_frame, text="150", width=5)
        canny2_value.grid(row=1, column=2, padx=(5, 0), pady=5)

        # Botones de acción
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(action_frame, text="Aplicar Bordes", command=self.finalize_edge_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Restaurar Original", command=self.reset_image).pack(side=tk.LEFT, padx=5)

        parent.columnconfigure(1, weight=1)

        # Actualizar etiquetas de valores
        def update_edge_values(*args):
            threshold_value.config(text=f"{int(self.threshold_var.get())}")
            canny1_value.config(text=f"{int(self.canny_threshold1_var.get())}")
            canny2_value.config(text=f"{int(self.canny_threshold2_var.get())}")

        self.threshold_var.trace('w', update_edge_values)
        self.canny_threshold1_var.trace('w', update_edge_values)
        self.canny_threshold2_var.trace('w', update_edge_values)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )

        if file_path:
            try:
                # Cargar imagen con PIL
                self.original_image = Image.open(file_path).convert('RGB')
                self.current_image = self.original_image.copy()
                self.filename = os.path.basename(file_path)

                # Reiniciar historial
                self.history = [self.current_image.copy()]
                self.history_index = 0
                self.zoom_factor = 1.0

                # Reiniciar controles
                self.reset_controls()

                # Mostrar imagen
                self.display_image_on_canvas()
                self.update_info()

                self.status_bar.config(text=f"PhotosCom - Imagen cargada: {self.filename}")

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
        self.threshold_var.set(100)
        self.canny_threshold1_var.set(50)
        self.canny_threshold2_var.set(150)
        self.kernel_size_var.set(3)

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
            # Aplicar zoom
            width, height = self.current_image.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)

            img = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convertir a PhotoImage para mostrar en tkinter
            self.display_image = ImageTk.PhotoImage(img)

            # Actualizar el label en el contenedor
            if hasattr(self, 'image_label'):
                self.image_label.config(image=self.display_image)
            else:
                self.image_label = ttk.Label(self.image_container, image=self.display_image)
                self.image_label.pack()

            # Actualizar región de scroll
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
                self.status_bar.config(text=f"PhotosCom - Imagen guardada: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen: {str(e)}")

    def add_to_history(self):
        # Si estamos en medio del historial, eliminar todo lo que está después
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]

        # Agregar nueva imagen al historial
        self.history.append(self.current_image.copy())
        self.history_index = len(self.history) - 1

        # Limitar el historial a 20 pasos para no consumir mucha memoria
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
            self.status_bar.config(text="PhotosCom - Deshacer: paso anterior")

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.update_info()
            self.status_bar.config(text="PhotosCom - Rehacer: paso siguiente")

    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.add_to_history()
            self.reset_controls()
            self.display_image_on_canvas()
            self.zoom_factor = 1.0
            self.status_bar.config(text="PhotosCom - Imagen restaurada a su estado original")

    def quick_rotate(self, angle):
        if not self.current_image:
            return

        img = self.history[self.history_index].copy()
        img = img.rotate(angle, expand=True)

        self.current_image = img
        self.add_to_history()
        self.display_image_on_canvas()
        self.status_bar.config(text=f"PhotosCom - Rotación aplicada: {angle}°")

    def preview_transform(self, transform_type):
        if not self.current_image:
            return

        # Para previsualización, trabajamos con la imagen base del historial
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

        # Aplicar transformaciones definitivas y añadir al historial
        img = self.history[self.history_index].copy()

        # Rotación
        angle = self.rotate_var.get()
        if angle != 0:
            img = img.rotate(angle, expand=True)

        # Escala
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
        self.status_bar.config(text="PhotosCom - Transformaciones aplicadas")

        # Reiniciar controles de transformación
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
            text=f"PhotosCom - Imagen volteada {'horizontalmente' if direction == 'horizontal' else 'verticalmente'}")

    def apply_adjustments(self):
        if not self.current_image:
            return

        # Para previsualización, trabajamos con la imagen base del historial
        img = self.history[self.history_index].copy()

        # Ajustar brillo
        brightness = self.brightness_var.get()
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        # Ajustar contraste
        contrast = self.contrast_var.get()
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)

        # Ajustar saturación
        saturation = self.saturation_var.get()
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)

        # Ajustar nitidez
        sharpen = self.sharpen_var.get()
        if sharpen != 1.0:
            # Aplicar filtro de nitidez
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        self.current_image = img
        self.display_image_on_canvas()

    def finalize_adjustments(self):
        """Aplica los ajustes permanentemente al historial"""
        if not self.current_image:
            return

        # Solo añadir al historial si realmente hay cambios
        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        saturation = self.saturation_var.get()
        sharpen = self.sharpen_var.get()

        if brightness != 1.0 or contrast != 1.0 or saturation != 1.0 or sharpen != 1.0:
            self.add_to_history()
            self.status_bar.config(text="PhotosCom - Ajustes aplicados permanentemente")

            # Reiniciar controles de ajuste
            self.brightness_var.set(1.0)
            self.contrast_var.set(1.0)
            self.saturation_var.set(1.0)
            self.sharpen_var.set(1.0)

    def apply_filter(self):
        if not self.current_image:
            return

        # Para filtros, siempre trabajamos con la imagen base del historial
        img = self.history[self.history_index].copy()
        filter_type = self.filter_var.get()

        if filter_type == "grayscale":
            img = img.convert("L").convert("RGB")
        elif filter_type == "sepia":
            # Convertir a numpy array para aplicar filtro sepia
            img_array = np.array(img)
            # Aplicar matriz de filtro sepia
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
            sepia_img = np.dot(img_array, sepia_filter.T)
            # Asegurar que los valores estén en el rango correcto
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            img = Image.fromarray(sepia_img)
        elif filter_type == "invert":
            # Invertir colores
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
        # Para "original" no hacemos nada, ya tenemos la imagen original

        self.current_image = img
        self.display_image_on_canvas()

        # Si no es el filtro original, añadir al historial
        if filter_type != "original":
            self.add_to_history()
            self.status_bar.config(text=f"PhotosCom - Filtro aplicado: {filter_type}")
        else:
            self.status_bar.config(text="PhotosCom - Filtro original restaurado")

    # NUEVOS MÉTODOS PARA DETECCIÓN DE BORDES

    def apply_edge_detection(self):
        """Aplica el operador de bordes seleccionado para previsualización"""
        if not self.current_image:
            return

        # Para previsualización, trabajamos con la imagen base del historial
        img = self.history[self.history_index].copy()
        edge_img = self.detect_edges(img)

        self.current_image = edge_img
        self.display_image_on_canvas()

    def finalize_edge_detection(self):
        """Aplica la detección de bordes permanentemente al historial"""
        if not self.current_image:
            return

        img = self.history[self.history_index].copy()
        edge_img = self.detect_edges(img)

        self.current_image = edge_img
        self.add_to_history()
        self.display_image_on_canvas()
        self.status_bar.config(text=f"PhotosCom - Detección de bordes aplicada: {self.edge_operator_var.get()}")

    def detect_edges(self, img):
        """Aplica el operador de detección de bordes seleccionado"""
        # Convertir a escala de grises si es necesario
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img

        # Convertir a array numpy para procesamiento
        img_array = np.array(img_gray)

        operator = self.edge_operator_var.get()
        threshold = self.threshold_var.get()
        kernel_size = self.kernel_size_var.get()

        # Aplicar el operador seleccionado
        if operator == "sobel":
            result = self.sobel_operator(img_array, threshold)
        elif operator == "prewitt":
            result = self.prewitt_operator(img_array, threshold)
        elif operator == "roberts":
            result = self.roberts_operator(img_array, threshold)
        elif operator == "kirsch":
            result = self.kirsch_operator(img_array, threshold)
        elif operator == "robinson":
            result = self.robinson_operator(img_array, threshold)
        elif operator == "frei_chen":
            result = self.frei_chen_operator(img_array, threshold)
        elif operator == "canny":
            result = self.canny_operator(img_array)
        elif operator == "gradient":
            result = self.gradient_operator(img_array, threshold)
        else:
            result = img_array

        # Convertir de vuelta a PIL Image
        return Image.fromarray(result)

    def sobel_operator(self, img, threshold):
        """Operador de Sobel"""
        # Aplicar filtro Sobel
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        # Calcular magnitud
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Normalizar y aplicar umbral
        magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
        _, result = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return result

    def prewitt_operator(self, img, threshold):
        """Operador de Prewitt"""
        # Definir kernels de Prewitt
        kernelx = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])
        kernely = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])

        # Aplicar convolución
        prewittx = cv2.filter2D(img, cv2.CV_64F, kernelx)
        prewitty = cv2.filter2D(img, cv2.CV_64F, kernely)

        # Calcular magnitud
        magnitude = np.sqrt(prewittx ** 2 + prewitty ** 2)

        # Normalizar y aplicar umbral
        magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
        _, result = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return result

    def roberts_operator(self, img, threshold):
        """Operador de Roberts"""
        # Definir kernels de Roberts
        kernelx = np.array([[1, 0],
                            [0, -1]])
        kernely = np.array([[0, 1],
                            [-1, 0]])

        # Aplicar convolución
        robertsx = cv2.filter2D(img, cv2.CV_64F, kernelx)
        robertsy = cv2.filter2D(img, cv2.CV_64F, kernely)

        # Calcular magnitud
        magnitude = np.sqrt(robertsx ** 2 + robertsy ** 2)

        # Normalizar y aplicar umbral
        magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
        _, result = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return result

    def kirsch_operator(self, img, threshold):
        """Operador de Kirsch (8 direcciones)"""
        # Definir kernels de Kirsch en 8 direcciones
        kernels = [
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # 0°
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # 45°
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # 90°
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # 135°
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # 180°
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # 225°
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # 270°
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])  # 315°
        ]

        # Aplicar todos los kernels y tomar el máximo
        responses = []
        for kernel in kernels:
            response = cv2.filter2D(img, cv2.CV_64F, kernel)
            responses.append(response)

        # Tomar el máximo de todas las respuestas
        magnitude = np.max(responses, axis=0)

        # Normalizar y aplicar umbral
        magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
        _, result = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return result

    def robinson_operator(self, img, threshold):
        """Operador de Robinson (8 direcciones)"""
        # Definir kernels de Robinson en 8 direcciones
        kernels = [
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # 0°
            np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),  # 45°
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),  # 90°
            np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),  # 135°
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),  # 180°
            np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),  # 225°
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # 270°
            np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])  # 315°
        ]

        # Aplicar todos los kernels y tomar el máximo
        responses = []
        for kernel in kernels:
            response = cv2.filter2D(img, cv2.CV_64F, kernel)
            responses.append(response)

        # Tomar el máximo de todas las respuestas
        magnitude = np.max(responses, axis=0)

        # Normalizar y aplicar umbral
        magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
        _, result = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return result

    def frei_chen_operator(self, img, threshold):
        """Operador de Frei-Chen"""
        # Definir máscaras de Frei-Chen (solo las primeras 4 para bordes)
        masks = [
            np.array([[1, np.sqrt(2), 1], [0, 0, 0], [-1, -np.sqrt(2), -1]]) / (2 * np.sqrt(2)),  # f1
            np.array([[1, 0, -1], [np.sqrt(2), 0, -np.sqrt(2)], [1, 0, -1]]) / (2 * np.sqrt(2)),  # f2
            np.array([[0, -1, np.sqrt(2)], [1, 0, -1], [-np.sqrt(2), 1, 0]]) / (2 * np.sqrt(2)),  # f3
            np.array([[np.sqrt(2), -1, 0], [-1, 0, 1], [0, 1, -np.sqrt(2)]]) / (2 * np.sqrt(2))  # f4
        ]

        # Aplicar todas las máscaras
        responses = []
        for mask in masks:
            response = cv2.filter2D(img, cv2.CV_64F, mask)
            responses.append(response ** 2)

        # Calcular magnitud del subespacio de bordes
        edge_space = np.sum(responses, axis=0)
        total_energy = np.sum(img.astype(np.float64) ** 2)  # Simplificación

        # Calcular proyección en subespacio de bordes
        if total_energy > 0:
            magnitude = np.sqrt(edge_space / total_energy) * 255
        else:
            magnitude = np.zeros_like(img, dtype=np.uint8)

        # Aplicar umbral
        magnitude = np.uint8(magnitude)
        _, result = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return result

    def canny_operator(self, img):
        """Algoritmo de Canny"""
        # Usar los umbrales específicos de Canny
        threshold1 = self.canny_threshold1_var.get()
        threshold2 = self.canny_threshold2_var.get()

        # Aplicar Canny
        edges = cv2.Canny(img, threshold1, threshold2)

        return edges

    def gradient_operator(self, img, threshold):
        """Operador de Gradiente básico"""
        # Calcular gradientes usando Sobel
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        # Calcular magnitud del gradiente
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Normalizar y aplicar umbral
        magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
        _, result = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)

        return result

    def zoom_in(self):
        if self.current_image:
            self.zoom_factor *= 1.2
            self.display_image_on_canvas()
            self.status_bar.config(text=f"PhotosCom - Zoom: {int(self.zoom_factor * 100)}%")

    def zoom_out(self):
        if self.current_image:
            self.zoom_factor /= 1.2
            self.display_image_on_canvas()
            self.status_bar.config(text=f"PhotosCom - Zoom: {int(self.zoom_factor * 100)}%")

    def zoom_fit(self):
        if self.current_image:
            self.zoom_factor = 1.0
            self.display_image_on_canvas()
            self.status_bar.config(text="PhotosCom - Zoom ajustado a la imagen")

    def on_container_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width, height=event.height)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_resize(self, event):
        # Solo redibujar si el evento es para el root window
        if event.widget == self.root:
            self.display_image_on_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoEditor(root)

    # Vincular evento de redimensionamiento
    root.bind("<Configure>", app.on_resize)

    root.mainloop()