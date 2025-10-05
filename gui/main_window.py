"""
Ventana principal de PhotoEscom
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import sys
sys.path.append('..')
from utils.constants import UI_COLORS, SUPPORTED_FORMATS, SAVE_FORMATS
from .toolbar import Toolbar
from .panels import (
    ToolsPanel,
    TransformPanel,
    AdjustmentsPanel,
    FiltersPanel,
    EdgeDetectionPanel
)
from .styles import setup_styles


class PhotoEditorGUI:
    """Clase principal de la interfaz gráfica"""

    def __init__(self, root, controller):
        """
        Inicializar GUI

        Args:
            root: Ventana raíz de Tkinter
            controller: Controlador principal que maneja la lógica
        """
        self.root = root
        self.controller = controller

        # Configurar ventana
        self.root.title("PhotoEscom - Editor de Fotos Profesional")
        self.root.geometry("1400x900")
        self.root.configure(bg=UI_COLORS['bg_color'])
        self.root.minsize(1200, 700)

        # Variables de la interfaz
        self.variables = {}
        self.init_variables()

        # Aplicar estilos
        self.style = setup_styles()

        # Widgets
        self.toolbar = None
        self.canvas = None
        self.image_label = None
        self.status_bar = None
        self.display_image = None

        # Paneles
        self.tools_panel = None
        self.transform_panel = None
        self.adjustments_panel = None
        self.filters_panel = None
        self.edge_panel = None

        # Crear interfaz
        self.create_widgets()

    def init_variables(self):
        """Inicializar variables de Tkinter"""
        # Transformaciones
        self.variables['rotate'] = tk.DoubleVar(value=0)
        self.variables['scale_x'] = tk.DoubleVar(value=1.0)
        self.variables['scale_y'] = tk.DoubleVar(value=1.0)

        # Ajustes
        self.variables['brightness'] = tk.DoubleVar(value=1.0)
        self.variables['contrast'] = tk.DoubleVar(value=1.0)
        self.variables['saturation'] = tk.DoubleVar(value=1.0)
        self.variables['sharpen'] = tk.DoubleVar(value=1.0)

        # Filtros
        self.variables['filter'] = tk.StringVar(value="original")

        # Detección de bordes
        self.variables['edge_operator'] = tk.StringVar(value="sobel")
        self.variables['threshold'] = tk.DoubleVar(value=30.0)
        self.variables['sigma'] = tk.DoubleVar(value=1.0)
        self.variables['canny_low'] = tk.DoubleVar(value=50.0)
        self.variables['canny_high'] = tk.DoubleVar(value=150.0)
        self.variables['roberts_form'] = tk.StringVar(value="sqrt")
        self.variables['show_magnitude'] = tk.BooleanVar(value=False)
        self.variables['show_angle'] = tk.BooleanVar(value=False)
        self.variables['extended_size'] = tk.IntVar(value=3)

    def create_widgets(self):
        """Crear todos los widgets de la interfaz"""
        # Callbacks para toolbar
        toolbar_callbacks = {
            'load': self.controller.load_image,
            'save': self.controller.save_image,
            'undo': self.controller.undo,
            'redo': self.controller.redo,
            'reset': self.controller.reset_image,
            'zoom_in': self.controller.zoom_in,
            'zoom_out': self.controller.zoom_out,
            'zoom_fit': self.controller.zoom_fit
        }

        # Crear toolbar
        self.toolbar = Toolbar(self.root, toolbar_callbacks)

        # Panel principal
        main_panel = ttk.Frame(self.root)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Notebook de herramientas (izquierda)
        tools_notebook = ttk.Notebook(main_panel, width=350)
        tools_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tools_notebook.pack_propagate(False)

        # Crear paneles
        self.create_tools_panels(tools_notebook)

        # Panel de visualización (derecha)
        self.image_frame = ttk.Frame(main_panel)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Crear canvas para imagen
        self.create_image_canvas()

        # Barra de estado
        self.status_bar = ttk.Label(
            self.root,
            text="PhotoEscom - Listo. Cargue una imagen para comenzar"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_tools_panels(self, notebook):
        """Crear paneles de herramientas"""
        # Callbacks para paneles
        tools_callbacks = {
            'flip': self.controller.apply_flip,
            'quick_rotate': self.controller.quick_rotate
        }

        transform_callbacks = {
            'preview_transform': self.controller.preview_transform,
            'apply_transforms': self.controller.apply_transforms
        }

        adjustments_callbacks = {
            'apply_adjustments': self.controller.apply_adjustments,
            'finalize_adjustments': self.controller.finalize_adjustments
        }

        filters_callbacks = {
            'apply_filter': self.controller.apply_filter
        }

        edge_callbacks = {
            'preview_edge': self.controller.preview_edge_detection,
            'apply_edge': self.controller.apply_edge_detection
        }

        # Panel de herramientas básicas
        self.tools_panel = ToolsPanel(
            notebook,
            tools_callbacks,
            self.variables
        )
        notebook.add(self.tools_panel.get_frame(), text="Herramientas")

        # Panel de transformaciones
        self.transform_panel = TransformPanel(
            notebook,
            transform_callbacks,
            self.variables
        )
        notebook.add(self.transform_panel.get_frame(), text="Transformar")

        # Panel de ajustes
        self.adjustments_panel = AdjustmentsPanel(
            notebook,
            adjustments_callbacks,
            self.variables
        )
        notebook.add(self.adjustments_panel.get_frame(), text="Ajustes")

        # Panel de filtros
        self.filters_panel = FiltersPanel(
            notebook,
            filters_callbacks,
            self.variables
        )
        notebook.add(self.filters_panel.get_frame(), text="Filtros")

        # Panel de detección de bordes
        edge_parent = ttk.Frame(notebook)
        self.edge_panel = EdgeDetectionPanel(
            edge_parent,
            edge_callbacks,
            self.variables
        )
        notebook.add(edge_parent, text="Detección Bordes")

    def create_image_canvas(self):
        """Crear canvas con scrollbars para la imagen"""
        canvas_container = ttk.Frame(self.image_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)

        # Canvas
        self.canvas = tk.Canvas(
            canvas_container,
            bg=UI_COLORS['canvas_bg'],
            highlightthickness=0,
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )

        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        v_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        h_scrollbar.grid(row=1, column=0, sticky=tk.EW)

        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        # Container para la imagen
        self.image_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.image_container,
            anchor="nw"
        )

        # Bindings
        self.image_container.bind(
            "<Configure>",
            self.on_container_configure
        )
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def display_image(self, pil_image, zoom_factor=1.0):
        """
        Mostrar imagen en el canvas

        Args:
            pil_image: PIL Image a mostrar
            zoom_factor: Factor de zoom
        """
        if pil_image is None:
            return

        width, height = pil_image.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        img = pil_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        self.display_image = ImageTk.PhotoImage(img)

        if self.image_label is None:
            self.image_label = ttk.Label(
                self.image_container,
                image=self.display_image
            )
            self.image_label.pack()
        else:
            self.image_label.config(image=self.display_image)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_info(self, info_text):
        """Actualizar información de la imagen"""
        self.tools_panel.update_info(info_text)

    def update_status(self, status_text):
        """Actualizar barra de estado"""
        self.status_bar.config(text=status_text)

    def reset_controls(self):
        """Resetear todos los controles a valores por defecto"""
        self.variables['rotate'].set(0)
        self.variables['scale_x'].set(1.0)
        self.variables['scale_y'].set(1.0)
        self.variables['brightness'].set(1.0)
        self.variables['contrast'].set(1.0)
        self.variables['saturation'].set(1.0)
        self.variables['sharpen'].set(1.0)
        self.variables['filter'].set("original")
        self.variables['edge_operator'].set("sobel")
        self.variables['threshold'].set(30.0)
        self.variables['sigma'].set(1.0)
        self.variables['canny_low'].set(50.0)
        self.variables['canny_high'].set(150.0)
        self.variables['roberts_form'].set("sqrt")
        self.variables['show_magnitude'].set(False)
        self.variables['show_angle'].set(False)
        self.variables['extended_size'].set(3)

    def show_error(self, title, message):
        """Mostrar mensaje de error"""
        messagebox.showerror(title, message)

    def show_warning(self, title, message):
        """Mostrar mensaje de advertencia"""
        messagebox.showwarning(title, message)

    def show_info(self, title, message):
        """Mostrar mensaje informativo"""
        messagebox.showinfo(title, message)

    def ask_yes_no(self, title, message):
        """Mostrar diálogo de confirmación"""
        return messagebox.askyesno(title, message)

    def ask_open_filename(self):
        """Abrir diálogo para seleccionar archivo"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=SUPPORTED_FORMATS
        )
        return filepath

    def ask_save_filename(self):
        """Abrir diálogo para guardar archivo"""
        filepath = filedialog.asksaveasfilename(
            title="Guardar imagen",
            defaultextension=".png",
            filetypes=SAVE_FORMATS
        )
        return filepath

    def on_container_configure(self, event):
        """Callback cuando el contenedor cambia de tamaño"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Callback cuando el canvas cambia de tamaño"""
        self.canvas.itemconfig(
            self.canvas_window,
            width=event.width,
            height=event.height
        )
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def get_variable(self, name):
        """Obtener variable por nombre"""
        return self.variables.get(name)