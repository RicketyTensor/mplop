import logging
import numpy
import typing
import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__file__)


class Figure(matplotlib.figure.Figure):
    """
    Specialized class inheriting from a basic matplotlib figure
    """

    def __init__(self, *args, nrows: int = 1, ncols: int = 1, sharex: bool = False, sharey: bool = False,
                 design: str = "single", title: str = None, **kwargs):
        """
        Constructor

        :param args: positional arguments
        :param nrows: number of rows of subplots
        :param ncols: number of columns fo subplots
        :param sharex: if a x axis should be shared among the subplots
        :param sharey: if an y axis should be shared among the subplots
        :param design: layout design of the subplots
        :param title: title of the figure (super title)
        :param kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)
        matplotlib.projections.register_projection(Axis)

        # Declare data
        self.ax = None  # Axes
        self.gs = None  # Grid Space

        # Process arguments
        if design == "single":
            if nrows != 1 or ncols != 1:
                logger.error(
                    "The argument combination of nrows={} and ncols={} and layout=single is not allows".format(nrows,
                                                                                                               ncols))
            self.init_fig_single()
        elif design == "grid":
            self.init_fig_grid(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey)
        else:
            logger.error("Not supported layout '{}' has been requested ".format(design))
            raise ValueError

        # Print title
        if title:
            self.suptitle(title)

        pass

    def init_fig_single(self) -> None:
        """
        Initialize a figure containing only a single axis

        :return:
        """
        self.gs = self.add_gridspec()
        self.ax = self.add_subplot(self.gs[0, 0], projection=Axis.name)

        pass

    def init_fig_grid(self, nrows: int = 1, ncols: int = 1, sharex: bool = False, sharey: bool = False) -> None:
        """
        Initialize a figure containing a grid of axes

        :param nrows: number of rows
        :param ncols: number of columns
        :param sharex: flag to share x axis
        :param sharey: flag to share y axis
        :return:
        """

        self.gs = self.add_gridspec(nrows=nrows, ncols=ncols)

        # Create first axis and set up linking
        axes = []
        axes.append(self.add_subplot(self.gs[0, 0], projection=Axis.name))
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
                    axes.append(self.add_subplot(self.gs[row, col], projection=Axis.name, **extra))
        self.ax = numpy.reshape(axes, (nrows, ncols))

        # Clear ticks for shared axes
        if sharex:
            for ax in self.ax[:nrows - 1, :].flat:
                ax.tick_params(labelbottom=False)
        if sharey:
            for ax in self.ax[:, 1:].flat:
                ax.tick_params(labelleft=False)

        pass

    def annotate_axes(self) -> None:
        if type(self.ax) == numpy.ndarray:
            for i, row in enumerate(self.ax):
                for j, ax in enumerate(row):
                    ax.text(0.5, 0.5, "{},{}".format(i, j), va="center", ha="center", fontsize=18, color="darkgrey")
                    ax.tick_params(labelbottom=False, labelleft=False)
        pass

    @classmethod
    def show(cls) -> None:
        """
        Show figure

        :return: None
        """
        pyplot.show()


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

    def heatmap(self, z, x=None, y=None, cbar: bool = True, cbarlabel: str = None, cmap: str = "YlGn",
                annotate: bool = False):
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

            texts = []
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    kw.update(color=textcolors[int(im.norm(z[i, j]) > threshold)])
                    text = im.axes.text(xr[0] + xs * j + xs / 2, yr[0] + ys * i + ys / 2, valfmt(z[i, j], None), **kw)
                    texts.append(text)
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
