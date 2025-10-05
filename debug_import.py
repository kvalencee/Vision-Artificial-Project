"""
Archivo de depuración para verificar imports
"""

import tkinter as tk
from tkinter import filedialog

# Test simple
root = tk.Tk()
root.withdraw()

print("Tipo de filedialog.askopenfilename:", type(filedialog.askopenfilename))
print("Es callable?:", callable(filedialog.askopenfilename))

# Intentar usarlo
try:
    filepath = filedialog.askopenfilename(
        title="Prueba",
        filetypes=[("Imágenes", "*.jpg *.png")]
    )
    print("Funciona! Archivo seleccionado:", filepath)
except Exception as e:
    print("ERROR:", e)
    import traceback
    traceback.print_exc()