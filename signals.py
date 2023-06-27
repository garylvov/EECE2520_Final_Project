import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

sp.init_printing(use_latex='mathjax')

DEFAULT_NUM_SIGNAL_SAMPLES = 10000
DEFAULT_SIGNAL_BOUNDS = (-10, 10)

DEFAULT_NUM_FOURIER_SAMPLES = 10000
DEFAULT_FOURIER_BOUNDS = (-10, 10)

class Signal():
    """
    Represents a two dimensional signal.
    """
    def __init__(self, signal=None, t_substitution=None, signal_bounds=DEFAULT_SIGNAL_BOUNDS, signal_num_samples=DEFAULT_NUM_SIGNAL_SAMPLES,
                 fourier_bounds = DEFAULT_FOURIER_BOUNDS, fourier_num_samples=DEFAULT_NUM_FOURIER_SAMPLES):
        
        self.t = sp.symbols("t")
        self.w = sp.symbols("\omega")
        self.signal_bounds = signal_bounds
        self.fourier_bounds = fourier_bounds

        if signal is None:
            raise AttributeError("A signal must be instantiated.")
        else:
            self.signal = signal
        if t_substitution is not None:
            self.signal = signal.subs(self.t, t_substitution)
            
        self.signal_latex_expr = sp.latex(self.signal)

        self.fourier_transform = self.find_fourier_transform()
        self.fourier_latex_expr = sp.latex(self.fourier_transform)
        
        x_func = sp.lambdify(self.t, self.signal, "numpy")
        self.t_val = np.linspace(self.signal_bounds[0], self.signal_bounds[1], signal_num_samples)
        self.discretized_signal = x_func(self.t_val)

        fourier_transform_func = sp.lambdify(self.w, self.fourier_transform, "numpy")
        self.freq_val = np.linspace(self.fourier_bounds[0], self.fourier_bounds[1], fourier_num_samples)
        self.discretized_fourier_transform = fourier_transform_func(self.freq_val)

    def find_fourier_transform(self):
        """
        Returns the analytical fourier transform of the signal.
        """
        transform = sp.fourier_transform(self.signal, self.t, self.w)
        if isinstance(transform, sp.FourierTransform):
            raise ValueError(f"Could not determine the fourier transform of the signal {self.signal}")
        return transform

    def fourier_series(self):
        pass

    def plot_discretized_signal(self, ax):
        ax.plot(self.t_val, self.discretized_signal, label=rf"${self.signal_latex_expr}$")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x(t)$")
        ax.set_title(rf"Discretized Signal with Bounds {self.signal_bounds} and {len(self.t_val)} Samples")

    def plot_discretized_fourier_transform(self, ax):
        ax.plot(self.freq_val, self.discretized_fourier_transform, label=rf"${self.fourier_latex_expr}$")
        ax.set_xlabel(rf"$\omega$")
        ax.set_ylabel(rf"$X(\omega)$")
        ax.set_title(rf"Discretized Fourier Transform with Bounds {self.fourier_bounds} and {len(self.freq_val)} Samples")

    def combine(self, other, operator):
        if isinstance(other, Signal):
            return Signal(signal=operator(self.signal, other.signal), signal_bounds=self.signal_bounds)
        elif isinstance(other, sp.Symbol) or isinstance(other, sp.Expr):
            return Signal(signal=operator(self.signal, other), signal_bounds=self.signal_bounds)
        elif isinstance(other, int) or isinstance(other, float):
            return Signal(signal=operator(self.signal, other), signal_bounds=self.signal_bounds)
        else:
            raise TypeError(
                "Signal operations are only defined for other Signals or scalars.")

    def __mul__(self, other):
        return self.combine(other, operator=lambda x, y: x * y)

    def __true__div__(self, other):
        return self.combine(other, operator=lambda x, y: x / y)

    def __add__(self, other):
        return self.combine(other, operator=lambda x, y: x + y)

    def __sub__(self, other):
        return self.combine(other, operator=lambda x, y: x - y)

    @staticmethod
    def convolve(f, g, t, lower_limit=-sp.oo, upper_limit=sp.oo):
        # https://codereview.stackexchange.com/questions/174538/implementing-convolution-using-sympy
        tau = sp.Symbol("__very_unlikely_name__", real=True)
        return sp.integrate(f.subs(t, tau) * g.subs(t, t - tau), 
                        (tau, lower_limit, upper_limit))

class UnitStep(Signal):
    def __init__(self, t_substitution=None, signal_bounds=DEFAULT_SIGNAL_BOUNDS, signal_num_samples=DEFAULT_NUM_SIGNAL_SAMPLES,
                 fourier_bounds = DEFAULT_FOURIER_BOUNDS, fourier_num_samples=DEFAULT_NUM_FOURIER_SAMPLES):
        t = sp.symbols("t")
        u = sp.Heaviside(t)
        super().__init__(signal=u, t_substitution=t_substitution,
                     signal_bounds=signal_bounds, signal_num_samples=signal_num_samples, 
                     fourier_bounds=fourier_bounds, fourier_num_samples=fourier_num_samples)


class RectangularPulse(Signal):
    def __init__(self, t_substitution=None, signal_bounds=DEFAULT_SIGNAL_BOUNDS, signal_num_samples=DEFAULT_NUM_SIGNAL_SAMPLES,
                 fourier_bounds = DEFAULT_FOURIER_BOUNDS, fourier_num_samples=DEFAULT_NUM_FOURIER_SAMPLES):
        t = sp.symbols("t")
        u = sp.Heaviside(t)
        p = u.subs(t, t + 0.5) - u.subs(t, t - 0.5)
        super().__init__(signal=p, t_substitution=t_substitution,
                     signal_bounds=signal_bounds, signal_num_samples=signal_num_samples, 
                     fourier_bounds=fourier_bounds, fourier_num_samples=fourier_num_samples)


class TriangularPulse(Signal):
    def __init__(self, t_substitution=None, signal_bounds=DEFAULT_SIGNAL_BOUNDS, signal_num_samples=DEFAULT_NUM_SIGNAL_SAMPLES,
                 fourier_bounds = DEFAULT_FOURIER_BOUNDS, fourier_num_samples=DEFAULT_NUM_FOURIER_SAMPLES):
        t = sp.symbols("t")
        u = sp.Heaviside(t)
        p = u.subs(t, t + 0.5) - u.subs(t, t - 0.5)
        p = 2 * p.subs(t, t)
        tri = Signal.convolve(p, p, t)
        super().__init__(signal=tri, t_substitution=t_substitution,
                     signal_bounds=signal_bounds, signal_num_samples=signal_num_samples, 
                     fourier_bounds=fourier_bounds, fourier_num_samples=fourier_num_samples)


class Ramp(Signal):
    def __init__(self, t_substitution=None, signal_bounds=DEFAULT_SIGNAL_BOUNDS, signal_num_samples=DEFAULT_NUM_SIGNAL_SAMPLES,
                 fourier_bounds = DEFAULT_FOURIER_BOUNDS, fourier_num_samples=DEFAULT_NUM_FOURIER_SAMPLES):
        t = sp.symbols("t")
        u = sp.Heaviside(t)
        r = t * u
        super().__init__(signal=r, t_substitution=t_substitution,
                     signal_bounds=signal_bounds, signal_num_samples=signal_num_samples, 
                     fourier_bounds=fourier_bounds, fourier_num_samples=fourier_num_samples)