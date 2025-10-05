"""
Módulo de detección de bordes
Implementación basada en el libro 'Visión por Computador: Imágenes Digitales y Aplicaciones'
"""

from .base_operator import EdgeOperator
from .first_derivative import (
    GradientOperator,
    SobelOperator,
    PrewittOperator,
    RobertsOperator
)
from .compass_operators import (
    KirschOperator,
    RobinsonOperator
)
from .frei_chen import FreiChenOperator
from .canny import CannyOperator
from .second_derivative import LaplacianOperator
from .extended_sobel import ExtendedSobelOperator

__all__ = [
    'EdgeOperator',
    'GradientOperator',
    'SobelOperator',
    'PrewittOperator',
    'RobertsOperator',
    'KirschOperator',
    'RobinsonOperator',
    'FreiChenOperator',
    'CannyOperator',
    'LaplacianOperator',
    'ExtendedSobelOperator'
]