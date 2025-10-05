"""
Estilos y configuración visual de la interfaz
"""

from tkinter import ttk
import sys

from utils.constants import UI_COLORS


def setup_styles():
    """
    Configurar estilos visuales para la aplicación

    Returns:
        ttk.Style configurado
    """
    style = ttk.Style()
    style.theme_use('clam')

    # Colores de la configuración
    bg_color = UI_COLORS['bg_color']
    frame_bg = UI_COLORS['frame_bg']
    button_bg = UI_COLORS['button_bg']
    accent_color = UI_COLORS['accent_color']
    text_color = UI_COLORS['text_color']
    highlight_color = UI_COLORS['highlight_color']

    # Configurar estilos de widgets
    style.configure('TFrame', background=frame_bg)

    style.configure('TLabel',
                    background=frame_bg,
                    foreground=text_color,
                    font=('Segoe UI', 9))

    style.configure('TButton',
                    background=button_bg,
                    foreground=text_color,
                    borderwidth=1,
                    focuscolor=accent_color,
                    font=('Segoe UI', 9))

    style.configure('TScale',
                    background=frame_bg,
                    troughcolor=accent_color)

    style.configure('TCheckbutton',
                    background=frame_bg,
                    foreground=text_color,
                    font=('Segoe UI', 9))

    style.configure('TRadiobutton',
                    background=frame_bg,
                    foreground=text_color,
                    font=('Segoe UI', 9))

    style.configure('TNotebook',
                    background=bg_color)

    style.configure('TNotebook.Tab',
                    background=button_bg,
                    foreground=text_color,
                    padding=[10, 5],
                    font=('Segoe UI', 9, 'bold'))

    style.configure('TLabelframe',
                    background=frame_bg,
                    foreground=text_color,
                    borderwidth=2,
                    relief='groove')

    style.configure('TLabelframe.Label',
                    background=frame_bg,
                    foreground=accent_color,
                    font=('Segoe UI', 10, 'bold'))

    style.configure('TEntry',
                    fieldbackground=button_bg,
                    foreground=text_color,
                    borderwidth=1)

    # Mapeos de estados
    style.map('TNotebook.Tab',
              background=[('selected', accent_color)])

    style.map('TButton',
              background=[('active', highlight_color),
                          ('pressed', accent_color)])

    style.map('TCheckbutton',
              background=[('active', frame_bg)])

    style.map('TRadiobutton',
              background=[('active', frame_bg)])

    return style


def get_title_font():
    """Obtener fuente para títulos"""
    return ('Segoe UI', 16, 'bold')


def get_header_font():
    """Obtener fuente para encabezados"""
    return ('Segoe UI', 12, 'bold')


def get_normal_font():
    """Obtener fuente normal"""
    return ('Segoe UI', 9)


def get_small_font():
    """Obtener fuente pequeña"""
    return ('Segoe UI', 8)


def get_mono_font():
    """Obtener fuente monoespaciada"""
    return ('Consolas', 9)