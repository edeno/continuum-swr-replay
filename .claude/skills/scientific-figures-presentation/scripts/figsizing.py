# figsizing.py
INCHES_PER_CM: float = 1 / 2.54
POINTS_PER_INCH: int = 72


def cm_to_inch(*cm: float) -> float:
    return tuple(c * INCHES_PER_CM for c in cm)


def inch_to_cm(*inch: float) -> float:
    return tuple(i * 2.54 for i in inch)


def pt_to_inch(pt: float) -> float:
    return pt / POINTS_PER_INCH


def inch_to_px(inch: float, dpi: int = 450) -> int:
    return int(inch * dpi)


def px_to_inch(px: int, dpi: int = 450) -> float:
    return px / dpi


def pixels_for_textsize(pt: float, dpi: int = 450) -> int:
    return pt_to_inch(pt) * dpi


def figure_size_from_panels(
    nrows: int,
    ncols: int,
    panel_width_cm: float = 4.0,
    panel_height_cm: float = 4.0,
    wspace_cm: float = 0.5,
    hspace_cm: float = 0.5,
) -> float:
    """Calculate the figure size in inches based on the number of rows and columns of panels,
    their dimensions, and the spacing between them.
    Returns a tuple of (width, height) in inches.
    """
    total_width_cm = ncols * panel_width_cm + (ncols - 1) * wspace_cm
    total_height_cm = nrows * panel_height_cm + (nrows - 1) * hspace_cm
    return cm_to_inch(total_width_cm, total_height_cm)
