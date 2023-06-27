import sympy as sp
from math import pi
from matplotlib import pyplot as plt

from signals import Signal as x
from signals import UnitStep as u
from signals import RectangularPulse as rect
from signals import TriangularPulse as tri
from signals import DiracDelta as delta
from signals import Ramp as r

t = sp.symbols('t')

signals_to_proc = [rect(t),
                   x(sp.sin(t) * (1/(pi * t))),
                   tri(t),
                   rect(t) * sp.exp(-t),
                   u(t) * sp.exp(-t),
                   rect(t + 1) * t**2,
                   x(sp.sin(t * .5 - 1)* t)]


for signal_to_proc in signals_to_proc:
    fig = plt.figure(figsize=(20, 20))

    original_plot = fig.add_subplot(2, 2, 1)
    original_plot.plot(signal_to_proc.t_val, signal_to_proc.discretized_signal, label=rf"${signal_to_proc.signal_latex_expr}$")
    original_plot.legend(fontsize="20")
    original_plot.set_xlabel(r"$t$")
    original_plot.set_ylabel(r"$x(t)$")
    original_plot.set_title(rf"Original Discretized Signal with Bounds {signal_to_proc.signal_bounds} and {len(signal_to_proc.t_val)} Samples")

    signal_plot = fig.add_subplot(2, 2, 3)
    fourier_plot = fig.add_subplot(2, 2, 2)
    error_plot = fig.add_subplot(2, 2, 4)
    
    for plot in [signal_plot, fourier_plot, original_plot]:
        plot.set_ylim(-2, 2)
        plot.set_xlim(-10, 10)
        plot.grid(True)

    signal_to_proc.plot_discretized_time_domain(signal_plot)
    signal_to_proc.plot_discretized_freq_domain(fourier_plot)
    signal_to_proc.plot_error_of_fft_reconstruction(error_plot)

    signal_plot.legend(fontsize="7")
    fourier_plot.legend(fontsize="10")
    error_plot.legend(fontsize="10")
    plt.show()