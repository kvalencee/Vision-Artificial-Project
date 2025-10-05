"""
Paneles laterales de herramientas
"""

import tkinter as tk
from tkinter import ttk
import sys

from utils.constants import (
    OPERATOR_INFO,
    EDGE_OPERATORS,
    BASIC_FILTERS,
    EXTENDED_MASK_SIZES
)


class ToolsPanel:
    """Panel de herramientas básicas"""

    def __init__(self, parent, callbacks, variables):
        self.parent = parent
        self.callbacks = callbacks
        self.vars = variables
        self.frame = ttk.Frame(parent, padding=10)
        self.create_panel()

    def create_panel(self):
        """Crear panel de herramientas básicas"""
        # Información de imagen
        info_frame = ttk.LabelFrame(self.frame, text="Información", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_label = ttk.Label(
            info_frame,
            text="Sin imagen cargada",
            justify=tk.LEFT
        )
        self.info_label.pack(anchor=tk.W)

        # Herramientas rápidas
        tools_frame = ttk.LabelFrame(self.frame, text="Herramientas", padding=10)
        tools_frame.pack(fill=tk.X, pady=(0, 10))

        # Botones de volteo
        flip_frame = ttk.Frame(tools_frame)
        flip_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            flip_frame,
            text="↔ Horizontal",
            command=lambda: self.callbacks['flip']('horizontal')
        ).pack(side=tk.LEFT, expand=True, padx=2)

        ttk.Button(
            flip_frame,
            text="↕ Vertical",
            command=lambda: self.callbacks['flip']('vertical')
        ).pack(side=tk.LEFT, expand=True, padx=2)

        # Botones de rotación rápida
        rotate_frame = ttk.Frame(tools_frame)
        rotate_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            rotate_frame,
            text="↺ 90°",
            command=lambda: self.callbacks['quick_rotate'](-90)
        ).pack(side=tk.LEFT, expand=True, padx=2)

        ttk.Button(
            rotate_frame,
            text="↻ 90°",
            command=lambda: self.callbacks['quick_rotate'](90)
        ).pack(side=tk.LEFT, expand=True, padx=2)

    def update_info(self, info_text):
        """Actualizar texto de información"""
        self.info_label.config(text=info_text)

    def get_frame(self):
        return self.frame


class TransformPanel:
    """Panel de transformaciones"""

    def __init__(self, parent, callbacks, variables):
        self.parent = parent
        self.callbacks = callbacks
        self.vars = variables
        self.frame = ttk.Frame(parent, padding=10)
        self.create_panel()

    def create_panel(self):
        """Crear panel de transformaciones"""
        # Rotación
        ttk.Label(self.frame, text="Rotación:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        rotate_scale = ttk.Scale(
            self.frame,
            from_=-180,
            to=180,
            variable=self.vars['rotate'],
            command=lambda e: self.callbacks['preview_transform']('rotate')
        )
        rotate_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        rotate_entry = ttk.Entry(
            self.frame,
            textvariable=self.vars['rotate'],
            width=5
        )
        rotate_entry.grid(row=0, column=2, padx=(5, 0), pady=5)

        # Escala X
        ttk.Label(self.frame, text="Escala X:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        scale_x_scale = ttk.Scale(
            self.frame,
            from_=0.1,
            to=5.0,
            variable=self.vars['scale_x'],
            command=lambda e: self.callbacks['preview_transform']('scale')
        )
        scale_x_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        scale_x_entry = ttk.Entry(
            self.frame,
            textvariable=self.vars['scale_x'],
            width=5
        )
        scale_x_entry.grid(row=1, column=2, padx=(5, 0), pady=5)

        # Escala Y
        ttk.Label(self.frame, text="Escala Y:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        scale_y_scale = ttk.Scale(
            self.frame,
            from_=0.1,
            to=5.0,
            variable=self.vars['scale_y'],
            command=lambda e: self.callbacks['preview_transform']('scale')
        )
        scale_y_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        scale_y_entry = ttk.Entry(
            self.frame,
            textvariable=self.vars['scale_y'],
            width=5
        )
        scale_y_entry.grid(row=2, column=2, padx=(5, 0), pady=5)

        # Botón aplicar
        ttk.Button(
            self.frame,
            text="Aplicar",
            command=self.callbacks['apply_transforms']
        ).grid(row=3, column=0, columnspan=3, pady=10)

        self.frame.columnconfigure(1, weight=1)

    def get_frame(self):
        return self.frame


class AdjustmentsPanel:
    """Panel de ajustes de imagen"""

    def __init__(self, parent, callbacks, variables):
        self.parent = parent
        self.callbacks = callbacks
        self.vars = variables
        self.frame = ttk.Frame(parent, padding=10)
        self.value_labels = []
        self.create_panel()

    def create_panel(self):
        """Crear panel de ajustes"""
        adjustments = [
            ("Brillo:", self.vars['brightness'], 0.1, 3.0),
            ("Contraste:", self.vars['contrast'], 0.1, 3.0),
            ("Saturación:", self.vars['saturation'], 0.0, 3.0),
            ("Nitidez:", self.vars['sharpen'], 1.0, 5.0)
        ]

        for i, (label, var, min_val, max_val) in enumerate(adjustments):
            ttk.Label(self.frame, text=label).grid(
                row=i, column=0, sticky=tk.W, pady=5
            )

            scale = ttk.Scale(
                self.frame,
                from_=min_val,
                to=max_val,
                variable=var,
                command=lambda e: self.callbacks['apply_adjustments']()
            )
            scale.grid(row=i, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

            value_label = ttk.Label(self.frame, text="1.0", width=5)
            value_label.grid(row=i, column=2, padx=(5, 0), pady=5)
            self.value_labels.append((var, value_label))

        ttk.Button(
            self.frame,
            text="Aplicar Ajustes",
            command=self.callbacks['finalize_adjustments']
        ).grid(row=len(adjustments), column=0, columnspan=3, pady=10)

        self.frame.columnconfigure(1, weight=1)

        # Actualizar valores
        for var, label in self.value_labels:
            var.trace('w', lambda *args: self.update_values())

    def update_values(self):
        """Actualizar etiquetas de valores"""
        for var, label in self.value_labels:
            label.config(text=f"{var.get():.1f}")

    def get_frame(self):
        return self.frame


class FiltersPanel:
    """Panel de filtros"""

    def __init__(self, parent, callbacks, variables):
        self.parent = parent
        self.callbacks = callbacks
        self.vars = variables
        self.frame = ttk.Frame(parent, padding=10)
        self.create_panel()

    def create_panel(self):
        """Crear panel de filtros"""
        for i, (value, text) in enumerate(BASIC_FILTERS.items()):
            ttk.Radiobutton(
                self.frame,
                text=text,
                value=value,
                variable=self.vars['filter'],
                command=self.callbacks['apply_filter']
            ).grid(row=i, column=0, sticky=tk.W, pady=2)

    def get_frame(self):
        return self.frame


class EdgeDetectionPanel:
    """Panel de detección de bordes"""

    def __init__(self, parent, callbacks, variables):
        self.parent = parent
        self.callbacks = callbacks
        self.vars = variables
        self.frame = None
        self.operator_info_label = None
        self.create_panel()

    def create_panel(self):
        """Crear panel con scroll para detección de bordes"""
        # Canvas con scrollbar
        canvas = tk.Canvas(self.parent, bg='#3c3c3c', highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self.parent,
            orient="vertical",
            command=canvas.yview
        )
        self.frame = ttk.Frame(canvas)

        self.frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Contenido del panel
        self.create_operators_section()
        self.create_parameters_section()
        self.create_visualization_section()
        self.create_extension_section()
        self.create_roberts_section()
        self.create_info_section()
        self.create_actions_section()

        # Mouse wheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)

    def create_operators_section(self):
        """Crear sección de operadores"""
        # Primera derivada
        first_deriv_frame = ttk.LabelFrame(
            self.frame,
            text="Primera Derivada",
            padding=10
        )
        first_deriv_frame.pack(fill=tk.X, pady=(0, 10))

        operators_first = [
            ("Gradiente Básico", "gradient"),
            ("Sobel", "sobel"),
            ("Prewitt", "prewitt"),
            ("Roberts", "roberts")
        ]

        for i, (text, value) in enumerate(operators_first):
            ttk.Radiobutton(
                first_deriv_frame,
                text=text,
                value=value,
                variable=self.vars['edge_operator']
            ).grid(row=i // 2, column=i % 2, sticky=tk.W, pady=2)

        # Compass
        compass_frame = ttk.LabelFrame(
            self.frame,
            text="Compass",
            padding=10
        )
        compass_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            compass_frame,
            text="Kirsch (8 dir)",
            value="kirsch",
            variable=self.vars['edge_operator']
        ).grid(row=0, column=0, sticky=tk.W, pady=2)

        ttk.Radiobutton(
            compass_frame,
            text="Robinson (8 dir)",
            value="robinson",
            variable=self.vars['edge_operator']
        ).grid(row=0, column=1, sticky=tk.W, pady=2)

        # Frei-Chen
        frei_frame = ttk.LabelFrame(
            self.frame,
            text="Frei-Chen",
            padding=10
        )
        frei_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            frei_frame,
            text="Frei-Chen (9 máscaras)",
            value="frei_chen",
            variable=self.vars['edge_operator']
        ).pack(anchor=tk.W)

        # Canny
        canny_frame = ttk.LabelFrame(
            self.frame,
            text="Canny",
            padding=10
        )
        canny_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            canny_frame,
            text="Canny (Óptimo)",
            value="canny",
            variable=self.vars['edge_operator']
        ).pack(anchor=tk.W)

        # Segunda derivada
        second_deriv_frame = ttk.LabelFrame(
            self.frame,
            text="Segunda Derivada",
            padding=10
        )
        second_deriv_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            second_deriv_frame,
            text="Laplaciano (LoG)",
            value="laplacian",
            variable=self.vars['edge_operator']
        ).pack(anchor=tk.W)

    def create_parameters_section(self):
        """Crear sección de parámetros"""
        params_frame = ttk.LabelFrame(
            self.frame,
            text="Parámetros",
            padding=10
        )
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Threshold
        ttk.Label(params_frame, text="Umbral T (Ec. 6.5):").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        threshold_scale = ttk.Scale(
            params_frame,
            from_=0,
            to=255,
            variable=self.vars['threshold']
        )
        threshold_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        self.threshold_value = ttk.Label(params_frame, text="30", width=5)
        self.threshold_value.grid(row=0, column=2, padx=(5, 0), pady=5)

        # Sigma
        ttk.Label(params_frame, text="Sigma (σ):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        sigma_scale = ttk.Scale(
            params_frame,
            from_=0.5,
            to=3.0,
            variable=self.vars['sigma']
        )
        sigma_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        self.sigma_value = ttk.Label(params_frame, text="1.0", width=5)
        self.sigma_value.grid(row=1, column=2, padx=(5, 0), pady=5)

        # Canny low
        ttk.Label(params_frame, text="Canny t1:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        canny_low_scale = ttk.Scale(
            params_frame,
            from_=0,
            to=255,
            variable=self.vars['canny_low']
        )
        canny_low_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        self.canny_low_value = ttk.Label(params_frame, text="50", width=5)
        self.canny_low_value.grid(row=2, column=2, padx=(5, 0), pady=5)

        # Canny high
        ttk.Label(params_frame, text="Canny t2:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        canny_high_scale = ttk.Scale(
            params_frame,
            from_=0,
            to=255,
            variable=self.vars['canny_high']
        )
        canny_high_scale.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=(5, 0))

        self.canny_high_value = ttk.Label(params_frame, text="150", width=5)
        self.canny_high_value.grid(row=3, column=2, padx=(5, 0), pady=5)

        params_frame.columnconfigure(1, weight=1)

        # Trace para actualizar valores
        self.vars['threshold'].trace('w', lambda *args: self.update_param_values())
        self.vars['sigma'].trace('w', lambda *args: self.update_param_values())
        self.vars['canny_low'].trace('w', lambda *args: self.update_param_values())
        self.vars['canny_high'].trace('w', lambda *args: self.update_param_values())

    def create_visualization_section(self):
        """Crear sección de visualización"""
        viz_frame = ttk.LabelFrame(
            self.frame,
            text="Visualización",
            padding=10
        )
        viz_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(
            viz_frame,
            text="Mostrar |G| (magnitud)",
            variable=self.vars['show_magnitude']
        ).pack(anchor=tk.W)

        ttk.Checkbutton(
            viz_frame,
            text="Mostrar θ (ángulo)",
            variable=self.vars['show_angle']
        ).pack(anchor=tk.W)

    def create_extension_section(self):
        """Crear sección de extensión"""
        ext_frame = ttk.LabelFrame(
            self.frame,
            text="Extensión",
            padding=10
        )
        ext_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ext_frame, text="Tamaño máscara:").pack(anchor=tk.W)

        for size in EXTENDED_MASK_SIZES[:3]:  # 3, 5, 7
            ttk.Radiobutton(
                ext_frame,
                text=f"{size}x{size}",
                value=size,
                variable=self.vars['extended_size']
            ).pack(anchor=tk.W)

    def create_roberts_section(self):
        """Crear sección de Roberts"""
        roberts_frame = ttk.LabelFrame(
            self.frame,
            text="Roberts",
            padding=10
        )
        roberts_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            roberts_frame,
            text="Forma 1: √(D1²+D2²)",
            value="sqrt",
            variable=self.vars['roberts_form']
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            roberts_frame,
            text="Forma 2: |D1|+|D2|",
            value="abs",
            variable=self.vars['roberts_form']
        ).pack(anchor=tk.W)

    def create_info_section(self):
        """Crear sección de información"""
        info_frame = ttk.LabelFrame(
            self.frame,
            text="Información",
            padding=10
        )
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.operator_info_label = ttk.Label(
            info_frame,
            text="Seleccione un operador",
            wraplength=300,
            justify=tk.LEFT
        )
        self.operator_info_label.pack(anchor=tk.W)

        # Trace para actualizar info
        self.vars['edge_operator'].trace('w', lambda *args: self.update_operator_info())

    def create_actions_section(self):
        """Crear sección de acciones"""
        action_frame = ttk.Frame(self.frame)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            action_frame,
            text="Vista Previa",
            command=self.callbacks['preview_edge']
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            action_frame,
            text="Aplicar",
            command=self.callbacks['apply_edge']
        ).pack(side=tk.LEFT)

    def update_param_values(self):
        """Actualizar valores de parámetros"""
        self.threshold_value.config(
            text=f"{self.vars['threshold'].get():.0f}"
        )
        self.sigma_value.config(
            text=f"{self.vars['sigma'].get():.1f}"
        )
        self.canny_low_value.config(
            text=f"{self.vars['canny_low'].get():.0f}"
        )
        self.canny_high_value.config(
            text=f"{self.vars['canny_high'].get():.0f}"
        )

    def update_operator_info(self):
        """Actualizar información del operador"""
        operator = self.vars['edge_operator'].get()
        info_text = OPERATOR_INFO.get(operator, "Información no disponible")
        self.operator_info_label.config(text=info_text)

    def get_frame(self):
        return self.frame