"""
Barra de herramientas superior de la aplicación
"""

import tkinter as tk
from tkinter import ttk


class Toolbar:
    """Barra de herramientas con botones principales"""

    def __init__(self, parent, callbacks):
        """
        Inicializar toolbar

        Args:
            parent: Widget padre
            callbacks: Diccionario con callbacks para cada acción
        """
        self.parent = parent
        self.callbacks = callbacks
        self.frame = None
        self.create_toolbar()

    def create_toolbar(self):
        """Crear la barra de herramientas"""
        self.frame = ttk.Frame(self.parent, height=50)
        self.frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        self.frame.pack_propagate(False)

        # Título
        title_label = ttk.Label(
            self.frame,
            text="PhotoEscom",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=(10, 20))

        # Botones de archivo
        btn_load = ttk.Button(
            self.frame,
            text="📁 Cargar",
            command=self.callbacks.get('load'),
            width=12
        )
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_save = ttk.Button(
            self.frame,
            text="💾 Guardar",
            command=self.callbacks.get('save'),
            width=12
        )
        btn_save.pack(side=tk.LEFT, padx=5)

        # Separador
        separator1 = ttk.Separator(self.frame, orient=tk.VERTICAL)
        separator1.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Botones de historial
        btn_undo = ttk.Button(
            self.frame,
            text="↶ Deshacer",
            command=self.callbacks.get('undo'),
            width=10
        )
        btn_undo.pack(side=tk.LEFT, padx=5)

        btn_redo = ttk.Button(
            self.frame,
            text="↷ Rehacer",
            command=self.callbacks.get('redo'),
            width=10
        )
        btn_redo.pack(side=tk.LEFT, padx=5)

        btn_reset = ttk.Button(
            self.frame,
            text="🔄 Restaurar",
            command=self.callbacks.get('reset'),
            width=12
        )
        btn_reset.pack(side=tk.LEFT, padx=5)

        # Separador
        separator2 = ttk.Separator(self.frame, orient=tk.VERTICAL)
        separator2.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Controles de zoom
        self.create_zoom_controls()

    def create_zoom_controls(self):
        """Crear controles de zoom"""
        zoom_frame = ttk.Frame(self.frame)
        zoom_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)

        btn_zoom_out = ttk.Button(
            zoom_frame,
            text="-",
            command=self.callbacks.get('zoom_out'),
            width=3
        )
        btn_zoom_out.pack(side=tk.LEFT, padx=2)

        btn_zoom_in = ttk.Button(
            zoom_frame,
            text="+",
            command=self.callbacks.get('zoom_in'),
            width=3
        )
        btn_zoom_in.pack(side=tk.LEFT, padx=2)

        btn_zoom_fit = ttk.Button(
            zoom_frame,
            text="🔍 Ajustar",
            command=self.callbacks.get('zoom_fit'),
            width=8
        )
        btn_zoom_fit.pack(side=tk.LEFT, padx=2)

    def get_frame(self):
        """Obtener el frame de la toolbar"""
        return self.frame