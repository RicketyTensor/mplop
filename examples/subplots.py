import numpy
import mplop
import matplotlib
import matplotlib.pyplot as pyplot


def grid_layout():
    # Grid layout numbering
    fig = mplop.figure(nrows=2, ncols=2, design="grid", sharex=False, sharey=False, title="Super title",
                       constrained_layout=True, num="Grid layout numbering")
    fig.show_index()

    # Sharing axes
    fig = mplop.figure(nrows=2, ncols=2, design="grid", sharex=True, sharey=True,
                       constrained_layout=True, num="Grid ax sharing")

    x = numpy.linspace(0, 1, 50) * numpy.pi
    y = numpy.sin(x)
    fig.ax[1, 0].plot(x, y)

    x = numpy.linspace(-1, 0, 50) * numpy.pi
    y = numpy.sin(x)
    fig.ax[0, 1].plot(x, y)

    fig.show()

    pass


if __name__ == "__main__":
    grid_layout()
