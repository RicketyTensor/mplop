import logging
import numpy
import typing
import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statistics import mean

logger = logging.getLogger(__file__)


class Figure(matplotlib.figure.Figure):
    """
    Specialized class inheriting from a basic matplotlib figure
    """

    def __init__(self, *args, nrows: int = 1, ncols: int = 1, aspect: str = None,
                 sharex: bool = False, sharey: bool = False, title: str = None, show_index: bool = False, **kwargs):
        """
        Constructor

        :param args: positional arguments
        :param nrows: number of rows of subplots
        :param ncols: number of columns fo subplots
        :param aspect: set aspect ratio for all axes
        :param sharex: if a x axis should be shared among the subplots
        :param sharey: if an y axis should be shared among the subplots
        :param title: title of the figure (super title)
        :param show_index: show indexing of subplots
        :param kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)
        matplotlib.projections.register_projection(Axis)
        subplot_kw = {}

        # Declare data
        self.ax = None  # Axes
        self.gs = None  # Grid Space
        self.sharex = sharex  # Flag for axis sharing
        self.sharey = sharey  # Flag for axis sharing

        # Aspect ratios
        if aspect:
            subplot_kw["aspect"] = aspect

        # Process arguments
        if nrows == 1 and ncols == 1:
            self.init_fig_single(**subplot_kw)
        elif nrows < 1 or ncols < 1:
            logger.error("Not supported grid with nrows={} and ncols={} has been requested ".format(nrows, ncols))
            raise ValueError
        elif nrows > 1 or ncols > 1:
            self.init_fig_grid(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, **subplot_kw)
        else:
            logger.error("Not supported grid with nrows={} and ncols={} has been requested ".format(nrows, ncols))
            raise ValueError

        # Print title
        if title:
            self.suptitle(title)

        # Show indexing
        if show_index:
            self.show_index()

        pass

    def init_fig_single(self, **kwargs) -> None:
        """
        Initialize a figure containing only a single axis

        :param kwargs: keyword arguments
        :return:
        """
        self.gs = self.add_gridspec()
        self.ax = self.add_subplot(self.gs[0, 0], projection=Axis.name, **kwargs)

        pass

    def init_fig_grid(self, nrows: int = 1, ncols: int = 1, sharex: bool = False, sharey: bool = False,
                      **kwargs) -> None:
        """
        Initialize a figure containing a grid of axes

        :param nrows: number of rows
        :param ncols: number of columns
        :param sharex: flag to share x axis
        :param sharey: flag to share y axis
        :param kwargs: keyword arguments
        :return:
        """

        self.gs = self.add_gridspec(nrows=nrows, ncols=ncols)

        # Create first axis and set up linking
        axes = []
        axes.append(self.add_subplot(self.gs[0, 0], projection=Axis.name, **kwargs))
        extra = {}
        if sharex:
            extra["sharex"] = axes[0]
        if sharey:
            extra["sharey"] = axes[0]

        # Generate axes
        for row in range(nrows):
            for col in range(ncols):
                if row == 0 and col == 0:
                    continue
                else:
                    axes.append(self.add_subplot(self.gs[row, col], projection=Axis.name, **extra, **kwargs))
        self.ax = numpy.reshape(axes, (nrows, ncols))

        # Clear ticks for shared axes
        if sharex:
            for ax in self.ax[:nrows - 1, :].flat:
                ax.tick_params(labelbottom=False)
        if sharey:
            for ax in self.ax[:, 1:].flat:
                ax.tick_params(labelleft=False)

        pass

    def show_index(self) -> None:
        """
        Show access indices on figures

        :return: None
        """
        if isinstance(self.ax, Axis):
            self.ax.text(0.5, 0.5, "ax", va="center", ha="center", fontsize=18, color="darkgrey")
        elif isinstance(self.ax, numpy.ndarray):
            for i, row in enumerate(self.ax):
                for j, ax in enumerate(row):
                    ax.text(0.5, 0.5, "ax[{},{}]".format(i, j), va="center", ha="center", fontsize=18, color="darkgrey")
                    # ax.tick_params(labelbottom=False, labelleft=False)
        pass

    def set_size(self, x, y) -> None:
        """
        Set size of the fiugre

        :param x: horizontal size in cm
        :param y: vertical size in cm
        :return:
        """
        cm2inch = lambda x: 0.393701 * x
        self.set_size_inches(cm2inch(x), cm2inch(y))
        pass

    def save(self, outfile="figure.png", dpi=150) -> None:
        """
        Save figure to a file

        :param outfile: path to the output file
        :param dpi: resolution
        """
        self.savefig(outfile, dpi=dpi)
        pass

    @classmethod
    def show(cls) -> None:
        """
        Show figure

        :return: None
        """
        pyplot.show()

    def get_ax(self, i):
        """
        Get axes by a flat index
        :param i: index
        :return:
        """
        return self.ax.flat[i]

    def format(self, xlabel=None, ylabel=None, **kwargs) -> None:
        """
        Consistently format all axes in the figure
        :param xlabel: label for x axis
        :param ylabel: label for y axis
        :param kwargs: keyowrd arguments
        :return:
        """
        # General formatting
        if isinstance(self.ax, Axis):
            self.ax.format(xlabel=xlabel, ylabel=ylabel, **kwargs)
        elif isinstance(self.ax, numpy.ndarray):
            for i, row in enumerate(self.ax):
                for j, ax in enumerate(row):
                    # x label
                    if self.sharex and i != self.ax.shape[0] - 1:
                        xl = None
                    else:
                        xl = xlabel

                    # y label
                    if self.sharey and j != 0:
                        yl = None
                    else:
                        yl = ylabel

                    ax.format(xlabel=xl, ylabel=yl, **kwargs)

        pass


class Axis(matplotlib.axes.Axes):
    name = 'mplopaxis'

    def __init__(self, *args, **kwargs):
        """
        Constructor

        :param self:
        :param args:
        :param kwargs:
        :return:
        """
        super().__init__(*args, **kwargs)
        pass

    def lineplot(self, x=None, y=None, **kwargs) -> None:
        """
        Make plot with lines

        :param args: positional arguments
        :param kwargs: keyword arguments
        :return:
        """
        self.plot(x, y, **kwargs)
        pass

    def heatmap(self, z, x=None, y=None, cbar: bool = True, cbarlabel: str = None, cmap: str = "YlGn",
                annotate: bool = False) -> None:
        """
        Plot a heatmap

        :param z: numerical data to fill the heatmap with
        :param x: x values related to z
        :param y: y values related to z
        :param cbar: if a colorbar should be added
        :param cbarlabel: label for the colorbar
        :param cmap: colormap
        :param annotate: if the heatmap should be annotated
        :return:
        """
        xr = range(z.shape[0]) if x is None else x
        yr = range(z.shape[1]) if y is None else y
        xs = (xr[-1] - xr[0]) / len(xr)
        ys = (yr[-1] - yr[0]) / len(yr)

        # Generate figure
        im = self.imshow(z, extent=[min(xr), max(xr), min(yr), max(yr)], cmap=cmap)

        # Add colorbar
        if cbar:
            divider = make_axes_locatable(self)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = self.figure.colorbar(im, cax=cax, orientation='vertical')
            cb.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Annotate
        if annotate:
            # Default alignment
            kw = dict(horizontalalignment="center",
                      verticalalignment="center")

            # Text formatter
            valfmt = matplotlib.ticker.StrMethodFormatter("{x:.1f}")

            # Normalize colors
            threshold = im.norm(z.max()) / 2.

            # Color options for the text
            textcolors = ("black", "white")

            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    kw.update(color=textcolors[int(im.norm(z[i, j]) > threshold)])
                    text = im.axes.text(xr[0] + xs * j + xs / 2, yr[0] + ys * i + ys / 2, valfmt(z[i, j], None), **kw)

        pass

    def heatmap_poly(self, polygons, values, vmin: float = None, vmax: float = None, label: str = None,
                     cbar: bool = True, cbarlabel: str = None, cmap: str = "jet", annotate: bool = False,
                     tooltips: bool = False) -> None:
        """
        Make a heatmap from polygons

        :param polygons: spatial definition of polygons
        :param values: values to assign to the polygons
        :param vmin: minimum value for color scaling
        :param vmax: maximum value for color scaling
        :param label: label to describe the data
        :param cbar: flag to show a colorbar
        :param cbarlabel: label for a colorbar
        :param cmap: colormap name
        :param annotate: if the heatmap should be annotated
        :param tooltips: show tooltips on hover
        :return: None
        """

        valmin = min(values) if vmin is None else vmin
        valmax = max(values) if vmax is None else vmax

        # Define colormap
        cmap = matplotlib.cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=valmin, vmax=valmax)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        self.colorbar = sm

        elems = []
        for i, poly in enumerate(polygons):
            elems.extend(self.fill(poly[0], poly[1], facecolor=cmap(norm(values[i])), edgecolor="white", linewidth=1))

        # Add colorbar
        if cbar:
            divider = make_axes_locatable(self)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = pyplot.colorbar(sm, cax=cax, use_gridspec=True, orientation='vertical', label=cbarlabel)

        # Annotate
        # --------
        if annotate:
            # Default alignment
            kw = dict(horizontalalignment="center",
                      verticalalignment="center")

            # Text formatter
            valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

            for i, poly in enumerate(polygons):
                kw.update(color="white")
                text = self.text(mean(poly[0]), mean(poly[1]), valfmt(values[i], None), **kw)

        # Tooltips
        # --------
        if tooltips:
            annot = self.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                  bbox=dict(boxstyle="round", fc="w"),
                                  arrowprops=dict(arrowstyle="-"))
            annot.set_visible(False)

            def update_annot(poly, idx):
                pos = poly.get_xy()
                annot.xy = (mean(pos[:, 0]), mean(pos[:, 1]))
                text = "{:.2f}".format(values[i])
                annot.set_text(text)
                annot.get_bbox_patch().set_facecolor("w")
                annot.get_bbox_patch().set_alpha(0.9)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == self:
                    for i, el in enumerate(elems):
                        cont, ind = el.contains(event)
                        if cont:
                            update_annot(el, i)
                            annot.set_visible(True)
                            self.figure.canvas.draw_idle()
                        else:
                            if vis:
                                annot.set_visible(False)
                                self.figure.canvas.draw_idle()

            self.figure.canvas.mpl_connect("motion_notify_event", hover)

        pass

    def format(self, xlabel=None, ylabel=None, title=None, linestyle='-', gridlines='both') -> None:
        # Grid
        if gridlines:
            self.grid(True, which='major', axis=gridlines, alpha=0.7, linestyle=linestyle, zorder=0)

        # Labels
        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)

        # Title
        self.set_title(title, pad=15)
        pass


def show():
    """
    Show figure

    :return: None
    """
    pyplot.show()


def figure(**kwargs):
    """
    Generate figure

    :param kwargs: keyword parameters
    :return:
    """
    return pyplot.figure(FigureClass=Figure, **kwargs)
