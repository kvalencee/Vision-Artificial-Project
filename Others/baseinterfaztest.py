import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import os


class PhotoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("PhotosCom - Editor de Fotos Profesional")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2e2e2e')

        # Variables para manejar la imagen
        self.original_image = None
        self.current_image = None
        self.display_image = None
        self.history = []
        self.history_index = -1
        self.filename = None

        # Variables para controles
        self.rotate_var = tk.DoubleVar(value=0)
        self.scale_x_var = tk.DoubleVar(value=1.0)
        self.scale_y_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.sharpen_var = tk.DoubleVar(value=1.0)
        self.filter_var = tk.StringVar(value="original")

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

        # Configurar estilos
        self.style.configure('TFrame', background=self.frame_bg)
        self.style.configure('TLabel', background=self.frame_bg, foreground=self.text_color)
        self.style.configure('TButton', background=self.button_bg, foreground=self.text_color,
                             borderwidth=1, focuscolor=self.accent_color)
        self.style.configure('TScale', background=self.frame_bg, troughcolor=self.accent_color)
        self.style.configure('TCheckbutton', background=self.frame_bg, foreground=self.text_color)
        self.style.configure('TRadiobutton', background=self.frame_bg, foreground=self.text_color)

        self.style.map('TButton', background=[('active', self.accent_color)])

    def create_widgets(self):
        # Panel principal
        main_panel = ttk.Frame(self.root)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel de herramientas izquierdo
        tools_frame = ttk.Frame(main_panel, width=200)
        tools_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tools_frame.pack_propagate(False)

        # Panel de visualización de imagen
        self.image_frame = ttk.Frame(main_panel)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Crear lienzo para la imagen
        self.canvas = tk.Canvas(self.image_frame, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Barra de herramientas superior
        self.create_toolbar(tools_frame)

        # Panel de transformaciones
        self.create_transform_panel(tools_frame)

        # Panel de ajustes
        self.create_adjustments_panel(tools_frame)

        # Panel de filtros
        self.create_filters_panel(tools_frame)

        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="PhotosCom - Listo. Cargue una imagen para comenzar")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_toolbar(self, parent):
        # Título de la aplicación
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 15))

        title_label = ttk.Label(title_frame, text="PhotosCom", font=("Arial", 16, "bold"))
        title_label.pack(pady=5)

        subtitle_label = ttk.Label(title_frame, text="Editor Profesional de Fotos", font=("Arial", 10))
        subtitle_label.pack(pady=(0, 10))

        # Botones de control
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        btn_load = ttk.Button(control_frame, text="Cargar Imagen", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        btn_save = ttk.Button(control_frame, text="Guardar", command=self.save_image)
        btn_save.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # Botones de navegación de historial
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        btn_undo = ttk.Button(nav_frame, text="◀", command=self.undo, width=3)
        btn_undo.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        btn_redo = ttk.Button(nav_frame, text="▶", command=self.redo, width=3)
        btn_redo.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        btn_reset = ttk.Button(nav_frame, text="Restaurar Original", command=self.reset_image)
        btn_reset.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

    def create_transform_panel(self, parent):
        transform_frame = ttk.LabelFrame(parent, text="Transformaciones", padding=10)
        transform_frame.pack(fill=tk.X, pady=(0, 10))

        # Rotación
        ttk.Label(transform_frame, text="Rotación:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rotate_var = tk.DoubleVar(value=0)
        rotate_scale = ttk.Scale(transform_frame, from_=-180, to=180, variable=self.rotate_var,
                                 command=lambda e: self.preview_transform('rotate'))
        rotate_scale.grid(row=0, column=1, sticky=tk.EW, pady=2)

        rotate_entry = ttk.Entry(transform_frame, textvariable=self.rotate_var, width=5)
        rotate_entry.grid(row=0, column=2, padx=(5, 0), pady=2)

        # Escala
        ttk.Label(transform_frame, text="Escala X:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.scale_x_var = tk.DoubleVar(value=1.0)
        scale_x_scale = ttk.Scale(transform_frame, from_=0.1, to=5.0, variable=self.scale_x_var,
                                  command=lambda e: self.preview_transform('scale'))
        scale_x_scale.grid(row=1, column=1, sticky=tk.EW, pady=2)

        scale_x_entry = ttk.Entry(transform_frame, textvariable=self.scale_x_var, width=5)
        scale_x_entry.grid(row=1, column=2, padx=(5, 0), pady=2)

        ttk.Label(transform_frame, text="Escala Y:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.scale_y_var = tk.DoubleVar(value=1.0)
        scale_y_scale = ttk.Scale(transform_frame, from_=0.1, to=5.0, variable=self.scale_y_var,
                                  command=lambda e: self.preview_transform('scale'))
        scale_y_scale.grid(row=2, column=1, sticky=tk.EW, pady=2)

        scale_y_entry = ttk.Entry(transform_frame, textvariable=self.scale_y_var, width=5)
        scale_y_entry.grid(row=2, column=2, padx=(5, 0), pady=2)

        # Voltear
        flip_frame = ttk.Frame(transform_frame)
        flip_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
        ttk.Button(flip_frame, text="Voltear H", command=lambda: self.apply_flip('horizontal')).pack(side=tk.LEFT,
                                                                                                     expand=True)
        ttk.Button(flip_frame, text="Voltear V", command=lambda: self.apply_flip('vertical')).pack(side=tk.LEFT,
                                                                                                   expand=True)

        # Aplicar transformaciones
        ttk.Button(transform_frame, text="Aplicar Transformaciones", command=self.apply_transforms).grid(row=4,
                                                                                                         column=0,
                                                                                                         columnspan=3,
                                                                                                         pady=5)

        transform_frame.columnconfigure(1, weight=1)

    def create_adjustments_panel(self, parent):
        adjust_frame = ttk.LabelFrame(parent, text="Ajustes", padding=10)
        adjust_frame.pack(fill=tk.X, pady=(0, 10))

        # Brillo
        ttk.Label(adjust_frame, text="Brillo:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.brightness_var = tk.DoubleVar(value=1.0)
        ttk.Scale(adjust_frame, from_=0.1, to=3.0, variable=self.brightness_var,
                  command=lambda e: self.apply_adjustments()).grid(row=0, column=1, sticky=tk.EW, pady=2)

        # Contraste
        ttk.Label(adjust_frame, text="Contraste:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.contrast_var = tk.DoubleVar(value=1.0)
        ttk.Scale(adjust_frame, from_=0.1, to=3.0, variable=self.contrast_var,
                  command=lambda e: self.apply_adjustments()).grid(row=1, column=1, sticky=tk.EW, pady=2)

        # Saturación
        ttk.Label(adjust_frame, text="Saturación:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.saturation_var = tk.DoubleVar(value=1.0)
        ttk.Scale(adjust_frame, from_=0.0, to=3.0, variable=self.saturation_var,
                  command=lambda e: self.apply_adjustments()).grid(row=2, column=1, sticky=tk.EW, pady=2)

        # Nitidez
        ttk.Label(adjust_frame, text="Nitidez:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.sharpen_var = tk.DoubleVar(value=1.0)
        ttk.Scale(adjust_frame, from_=1.0, to=5.0, variable=self.sharpen_var,
                  command=lambda e: self.apply_adjustments()).grid(row=3, column=1, sticky=tk.EW, pady=2)

        # Botón para aplicar ajustes permanentemente
        ttk.Button(adjust_frame, text="Aplicar Ajustes", command=self.finalize_adjustments).grid(row=4, column=0,
                                                                                                 columnspan=2, pady=5)

        adjust_frame.columnconfigure(1, weight=1)

    def create_filters_panel(self, parent):
        filter_frame = ttk.LabelFrame(parent, text="Filtros", padding=10)
        filter_frame.pack(fill=tk.X)

        filters = [
            ("Original", "original"),
            ("Escala de Grises", "grayscale"),
            ("Sepia", "sepia"),
            ("Invertir", "invert"),
            ("Blur", "blur"),
            ("Detalle", "detail"),
            ("Bordes", "edges")
        ]

        self.filter_var = tk.StringVar(value="original")

        for text, mode in filters:
            ttk.Radiobutton(filter_frame, text=text, value=mode,
                            variable=self.filter_var, command=self.apply_filter).pack(anchor=tk.W)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff")]
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

                # Reiniciar controles
                self.reset_controls()

                # Mostrar imagen
                self.display_image_on_canvas()

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

    def display_image_on_canvas(self):
        if self.current_image:
            # Obtener dimensiones del canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                # Si el canvas es demasiado pequeño, esperar a que se redimensione
                self.root.after(100, self.display_image_on_canvas)
                return

            # Redimensionar imagen para que quepa en el canvas manteniendo la relación de aspecto
            img = self.current_image.copy()
            img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            # Convertir a PhotoImage para mostrar en tkinter
            self.display_image = ImageTk.PhotoImage(img)

            # Limpiar canvas y mostrar imagen
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                anchor=tk.CENTER,
                image=self.display_image
            )

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

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.status_bar.config(text="PhotosCom - Deshacer: paso anterior")

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.status_bar.config(text="PhotosCom - Rehacer: paso siguiente")

    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.add_to_history()
            self.reset_controls()
            self.display_image_on_canvas()
            self.status_bar.config(text="PhotosCom - Imagen restaurada a su estado original")

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
        # Para "original" no hacemos nada, ya tenemos la imagen original

        self.current_image = img
        self.display_image_on_canvas()

        # Si no es el filtro original, añadir al historial
        if filter_type != "original":
            self.add_to_history()
            self.status_bar.config(text=f"PhotosCom - Filtro aplicado: {filter_type}")
        else:
            self.status_bar.config(text="PhotosCom - Filtro original restaurado")

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