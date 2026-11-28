"""Image encoding and processing."""

from .bitplanes import encode_bitplanes, encode_gray4_bitplanes
from .compression import (
    DEFAULT_ZLIB_WINDOW_BITS,
    FIRMWARE_ZLIB_WINDOW_BITS,
    ZIPXL_ZLIB_WINDOW_BITS,
    compress_image_data,
    decompress_image_data,
    zlib_window_bits,
)
from .images import encode_1bpp, encode_2bpp, encode_4bpp, encode_image, fit_image, normalize_source_image
from .layout import (
    FrameGeometry,
    FrameLayout,
    FrameLayoutError,
    frame_size,
    lay_out_plane,
    lay_out_planes,
    resolve_layout,
    scheme_geometry,
)

__all__ = [
    "fit_image",
    "normalize_source_image",
    "encode_image",
    "encode_1bpp",
    "encode_2bpp",
    "encode_4bpp",
    "encode_bitplanes",
    "encode_gray4_bitplanes",
    "DEFAULT_ZLIB_WINDOW_BITS",
    "FIRMWARE_ZLIB_WINDOW_BITS",
    "ZIPXL_ZLIB_WINDOW_BITS",
    "compress_image_data",
    "decompress_image_data",
    "zlib_window_bits",
    "FrameGeometry",
    "FrameLayout",
    "FrameLayoutError",
    "frame_size",
    "lay_out_plane",
    "lay_out_planes",
    "resolve_layout",
    "scheme_geometry",
]
