"""
Constantes del proyecto PhotoEscom
Basado en el libro 'Visión por Computador: Imágenes Digitales y Aplicaciones'
"""

# Colores de la interfaz
UI_COLORS = {
    'bg_color': '#2e2e2e',
    'frame_bg': '#3c3c3c',
    'button_bg': '#4a4a4a',
    'accent_color': '#007acc',
    'text_color': '#ffffff',
    'highlight_color': '#4a76cf',
    'canvas_bg': '#1e1e1e'
}

# Formatos de imagen soportados
SUPPORTED_FORMATS = [
    ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")
]

SAVE_FORMATS = [
    ("PNG", "*.png"),
    ("JPEG", "*.jpg"),
    ("Todos", "*.*")
]

# Límites de historial
MAX_HISTORY_SIZE = 20

# Límites de parámetros
ROTATION_RANGE = (-180, 180)
SCALE_RANGE = (0.1, 5.0)
BRIGHTNESS_RANGE = (0.1, 3.0)
CONTRAST_RANGE = (0.1, 3.0)
SATURATION_RANGE = (0.0, 3.0)
SHARPEN_RANGE = (1.0, 5.0)

# Parámetros de detección de bordes
EDGE_THRESHOLD_RANGE = (0, 255)
SIGMA_RANGE = (0.5, 3.0)
CANNY_THRESHOLD_RANGE = (0, 255)

# Tamaños de máscaras extendidas disponibles
EXTENDED_MASK_SIZES = [3, 5, 7, 9, 11]

# Operadores de detección de bordes disponibles
EDGE_OPERATORS = {
    'gradient': 'Gradiente Básico',
    'sobel': 'Sobel',
    'prewitt': 'Prewitt',
    'roberts': 'Roberts',
    'kirsch': 'Kirsch',
    'robinson': 'Robinson',
    'frei_chen': 'Frei-Chen',
    'canny': 'Canny',
    'laplacian': 'Laplaciano'
}

# Información de operadores (del PDF)
OPERATOR_INFO = {
    'gradient': "Gradiente Básico: Diferencias finitas. Rápido pero sensible al ruido.",
    'sobel': "Sobel: Balance óptimo entre precisión y suavizado. Ec. (6.6)",
    'prewitt': "Prewitt: Similar a Sobel con coeficientes uniformes.",
    'roberts': "Roberts: Operador diagonal 2x2. Ec. (6.7) y (6.8)",
    'kirsch': "Kirsch: 8 máscaras direccionales. Toma máximo. Fig. 6.11",
    'robinson': "Robinson: Similar a Kirsch, máscara inicial diferente. Fig. 6.13",
    'frei_chen': "Frei-Chen: 9 máscaras - base vectorial completa. Ec. (6.10), (6.13)",
    'canny': "Canny: Algoritmo óptimo. 3 módulos: gradiente, supresión no máxima, histéresis.",
    'laplacian': "Laplaciano: Segunda derivada. Detecta zero-crossings."
}

# Filtros básicos
BASIC_FILTERS = {
    'original': 'Original',
    'grayscale': 'Escala de Grises',
    'sepia': 'Sepia',
    'invert': 'Invertir',
    'blur': 'Desenfoque',
    'detail': 'Detalle',
    'edges': 'Bordes',
    'enhance': 'Realce'
}

# Ecuaciones del libro (para referencia)
EQUATIONS = {
    'threshold': '(6.5)',  # g(x,y) = {1 si G[f(x,y)] > T, 0 si G[f(x,y)] ≤ T}
    'sobel': '(6.6)',      # Gx = (z3 + 2z6 + z9) - (z1 + 2z4 + z7)
    'roberts_1': '(6.7)',  # D1 = f(x,y) - f(x-1,y-1)
    'roberts_2': '(6.8)',  # R = sqrt(D1² + D2²) o R = |D1| + |D2|
    'frei_chen_proj': '(6.10)',  # R = suma(wi*zi)
    'frei_chen_cos': '(6.13)'    # cos(θ) = sqrt(M/S)
}

# Tamaño máximo para advertencia
MAX_IMAGE_SIZE = 10000000  # 10 megapixels

# Configuración de zoom
ZOOM_STEP = 1.2
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0