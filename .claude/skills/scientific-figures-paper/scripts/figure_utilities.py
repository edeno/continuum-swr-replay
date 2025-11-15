import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import sqrt

# Figure Parameters
MM_TO_INCHES = 1.0 / 25.4

ONE_AND_HALF_COLUMN = 140.0 * MM_TO_INCHES
TWO_COLUMN = 178.0 * MM_TO_INCHES
ONE_COLUMN = TWO_COLUMN / 2.0
PAGE_HEIGHT = 247.0 * MM_TO_INCHES
GOLDEN_RATIO = (sqrt(5) - 1.0) / 2.0


def set_figure_defaults(context="paper", style_path=None):
    """
    Set figure defaults for publication-quality plots.
    
    Parameters
    ----------
    context : str, default="paper"
        Context for the figure style. If "paper", loads paper.mplstyle.
        Otherwise, uses seaborn defaults.
    style_path : str, optional
        Custom path to .mplstyle file. If None, looks for paper.mplstyle
        in the skill's assets directory.
    """
    if context == "paper":
        if style_path is None:
            # Try to find paper.mplstyle in common locations
            possible_paths = [
                "assets/paper.mplstyle",
                "../assets/paper.mplstyle",
                "../../assets/paper.mplstyle",
                "./paper.mplstyle",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    style_path = path
                    break
        
        if style_path and os.path.exists(style_path):
            mpl.style.use(style_path)
        else:
            # Fallback to manual settings if style file not found
            rc_params = {
                "font.family": "Arial",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "axes.labelcolor": "black",
                "text.color": "black",
                "text.usetex": False,
                "figure.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.linewidth": 0.8,
                "axes.labelsize": 7,
                "axes.labelpad": 4,
                "axes.titlesize": 7,
                "axes.grid": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "xtick.labelsize": 5,
                "ytick.labelsize": 5,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.size": 3,
                "ytick.major.size": 3,
                "lines.linewidth": 1.5,
                "lines.markersize": 4,
                "legend.frameon": False,
                "legend.fontsize": 5,
                "savefig.dpi": 450,
                "savefig.format": "pdf",
                "savefig.transparent": False,
            }
            mpl.rcParams.update(rc_params)
    else:
        # Use seaborn defaults for presentations or other contexts
        rc_params = {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "text.usetex": False,
            "xtick.major.size": 5,
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.major.size": 5,
            "axes.labelpad": 0.1,
        }
        sns.set_theme(style="white", context=context, rc=rc_params, font_scale=0.6)


def save_figure(figure_name, output_dir=None, facecolor=None, transparent=True):
    """
    Save figure in both PDF and PNG formats.
    
    Parameters
    ----------
    figure_name : str
        Name of the figure (without extension)
    output_dir : str, optional
        Directory to save figures. If None, saves in current directory.
    facecolor : str, optional
        Background color for the figure
    transparent : bool, default=True
        Whether to save with transparent background
    """
    if output_dir is None:
        output_dir = "."
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_path = os.path.join(output_dir, f"{figure_name}.pdf")
    png_path = os.path.join(output_dir, f"{figure_name}.png")
    
    save_kwargs = {
        "transparent": transparent,
        "dpi": 300,
        "bbox_inches": "tight",
    }
    
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor
    
    plt.savefig(pdf_path, **save_kwargs)
    plt.savefig(png_path, **save_kwargs)
    
    print(f"✓ Saved: {pdf_path} and {png_path}")
