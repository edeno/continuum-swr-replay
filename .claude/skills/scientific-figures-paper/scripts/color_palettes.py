"""
Colorblind-friendly color palettes for scientific figures.

These palettes are designed to be:
- Distinguishable by people with color vision deficiencies
- Effective in grayscale
- Appropriate for scientific publications and presentations
"""

# Wong palette (most commonly used, 8 colors)
# Optimized for colorblind accessibility
# Reference: Bang Wong (2011) Nature Methods
WONG = [
    '#000000',  # Black
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion (Red-Orange)
    '#CC79A7',  # Reddish Purple
]

# Okabe-Ito palette (alternative colorblind-friendly palette)
# Reference: Okabe and Ito (2008)
OKABE_ITO = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
    '#000000',  # Black
]

# IBM Design Library palette (alternative with more colors)
IBM_COLORBLIND_SAFE = [
    '#648FFF',  # Blue
    '#785EF0',  # Purple
    '#DC267F',  # Magenta
    '#FE6100',  # Orange
    '#FFB000',  # Yellow
]

# Tol bright palette (high contrast, good for presentations)
TOL_BRIGHT = [
    '#4477AA',  # Blue
    '#EE6677',  # Red
    '#228833',  # Green
    '#CCBB44',  # Yellow
    '#66CCEE',  # Cyan
    '#AA3377',  # Purple
    '#BBBBBB',  # Grey
]

# Tol muted palette (subtle, good for papers)
TOL_MUTED = [
    '#332288',  # Indigo
    '#88CCEE',  # Cyan
    '#44AA99',  # Teal
    '#117733',  # Green
    '#999933',  # Olive
    '#DDCC77',  # Sand
    '#CC6677',  # Rose
    '#882255',  # Wine
    '#AA4499',  # Purple
]


def get_palette(name='wong'):
    """
    Get a color palette by name.
    
    Parameters
    ----------
    name : str, default='wong'
        Name of the palette. Options: 'wong', 'okabe_ito', 'ibm', 
        'tol_bright', 'tol_muted'
    
    Returns
    -------
    list
        List of hex color codes
    
    Examples
    --------
    >>> colors = get_palette('wong')
    >>> ax.plot(x, y, color=colors[1])  # Orange
    """
    palettes = {
        'wong': WONG,
        'okabe_ito': OKABE_ITO,
        'ibm': IBM_COLORBLIND_SAFE,
        'tol_bright': TOL_BRIGHT,
        'tol_muted': TOL_MUTED,
    }
    
    if name.lower() not in palettes:
        raise ValueError(f"Unknown palette '{name}'. "
                        f"Options: {list(palettes.keys())}")
    
    return palettes[name.lower()]


def show_palette(palette_name='wong', figsize=(8, 1)):
    """
    Display a color palette visually.
    
    Parameters
    ----------
    palette_name : str, default='wong'
        Name of palette to display
    figsize : tuple, default=(8, 1)
        Figure size in inches
    
    Examples
    --------
    >>> show_palette('wong')
    >>> show_palette('tol_bright')
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    colors = get_palette(palette_name)
    n_colors = len(colors)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, n_colors)
    ax.set_ylim(0, 1)
    
    for i, color in enumerate(colors):
        rect = mpatches.Rectangle((i, 0), 1, 1, facecolor=color, 
                                   edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, color, ha='center', va='center',
                fontsize=8, color='white' if i == 0 else 'black')
    
    ax.set_title(f'{palette_name.upper()} Palette', fontsize=12, pad=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# Helper functions for common use cases
def get_categorical_colors(n, palette='wong'):
    """
    Get n colors for categorical data.
    
    Parameters
    ----------
    n : int
        Number of colors needed
    palette : str, default='wong'
        Name of palette to use
    
    Returns
    -------
    list
        List of n hex color codes
    
    Examples
    --------
    >>> colors = get_categorical_colors(3)
    >>> for i, color in enumerate(colors):
    ...     ax.plot(x, y[i], color=color)
    """
    colors = get_palette(palette)
    if n > len(colors):
        raise ValueError(f"Palette '{palette}' only has {len(colors)} colors. "
                        f"You requested {n}.")
    return colors[:n]


def get_sequential_colormap(name='viridis'):
    """
    Get a perceptually uniform sequential colormap.
    
    Parameters
    ----------
    name : str, default='viridis'
        Colormap name. Options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    
    Returns
    -------
    matplotlib colormap
    
    Examples
    --------
    >>> cmap = get_sequential_colormap('plasma')
    >>> im = ax.imshow(data, cmap=cmap)
    """
    import matplotlib.pyplot as plt
    return plt.get_cmap(name)


if __name__ == '__main__':
    # Show all available palettes
    print("Colorblind-friendly palettes:")
    print("\nWong palette (8 colors):")
    show_palette('wong')
    
    print("\nOkabe-Ito palette (8 colors):")
    show_palette('okabe_ito')
    
    print("\nTol Bright palette (7 colors, high contrast):")
    show_palette('tol_bright')
    
    print("\nTol Muted palette (9 colors, subtle):")
    show_palette('tol_muted', figsize=(10, 1))
