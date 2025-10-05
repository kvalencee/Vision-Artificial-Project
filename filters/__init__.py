"""
Módulo de filtros y ajustes de imagen
"""

from .basic_filters import (
    apply_grayscale,
    apply_sepia,
    apply_invert,
    apply_basic_filter
)

from .adjustments import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    apply_sharpen,
    apply_all_adjustments
)

from .convolution_filters import (
    apply_blur,
    apply_detail,
    apply_edges,
    apply_enhance
)

__all__ = [
    # Basic filters
    'apply_grayscale',
    'apply_sepia',
    'apply_invert',
    'apply_basic_filter',

    # Adjustments
    'adjust_brightness',
    'adjust_contrast',
    'adjust_saturation',
    'apply_sharpen',
    'apply_all_adjustments',

    # Convolution filters
    'apply_blur',
    'apply_detail',
    'apply_edges',
    'apply_enhance'
]