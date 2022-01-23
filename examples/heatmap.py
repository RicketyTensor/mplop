import numpy
import mplop
import matplotlib
import matplotlib.pyplot as pyplot


def paraboloid(x, y):
    z = 1 - (x ** 2 + y ** 2)
    return z


def single_heatmap():
    # Grid layout numbering
    fig = mplop.figure(nrows=1, ncols=1, design="single", sharex=False, sharey=False,
                       constrained_layout=True, num="Single heatmap")

    # Generate data
    n = 5
    x, y = numpy.meshgrid(numpy.linspace(-1.0, 1.0, n), numpy.linspace(-1.0, 1.0, n))
    z = paraboloid(x, y)

    # Fill plot
    fig.ax.heatmap(z, x=x[0, :], y=y[:, 0], annotate=True)

    pass


if __name__ == "__main__":
    single_heatmap()
    mplop.show()
