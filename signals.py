import numpy as np
import sympy as sp
from scipy import fftpack
sp.init_printing(use_latex='mathjax')

DEFAULT_NUM_SIGNAL_SAMPLES = 10000
DEFAULT_SIGNAL_BOUNDS = (-50, 50)

DEFAULT_NUM_FOURIER_SAMPLES = 10000
DEFAULT_FOURIER_BOUNDS = (-50, 50)

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
        self.x_func = sp.lambdify(self.t, self.signal, "numpy")
        self.t_val = np.linspace(self.signal_bounds[0], self.signal_bounds[1], signal_num_samples)
        self.discretized_signal = self.x_func(self.t_val)

        self.freq_val = np.linspace(self.fourier_bounds[0], self.fourier_bounds[1], fourier_num_samples)
        try:
            self.fourier_transform = self.find_fourier_transform()
            self.fourier_latex_expr = sp.latex(self.fourier_transform)
            fourier_transform_func = sp.lambdify(self.w, self.fourier_transform, "numpy")
            self.discretized_fourier_transform = fourier_transform_func(self.freq_val)
        except ValueError:
            self.discretized_fourier_transform = None

        self.freq = np.fft.fftfreq(self.discretized_signal.shape[-1])
        self.fft_signal_freq = fftpack.fft(self.discretized_signal)

        self.num_terms = [3, 5, 20, 50, 200, 500, 2000, 5000]
        reconstructed_signal = [fftpack.ifft(self.fft_signal_freq, n=n) for n in self.num_terms]
        self.expansions = [np.real(signal) for signal in reconstructed_signal]

    def find_fourier_transform(self):
        """
        Returns the analytical fourier transform of the signal.
        """
        transform = sp.fourier_transform(self.signal, self.t, self.w)
        if isinstance(transform, sp.FourierTransform) or transform == 0:
            raise ValueError(f"Could not determine the fourier transform of the signal {self.signal}")
        return transform

    def plot_discretized_time_domain(self, ax):
        # it's peicewise if the latex expr. contains begin, need to fall back to normal rendering
        has_begin = self.signal_latex_expr.find("begin") != -1

        if has_begin:
            ax.plot(self.t_val, self.discretized_signal, label=f"Signal: {sp.pretty(self.signal)}")
        else:
            ax.plot(self.t_val, self.discretized_signal, label=rf"Signal: ${self.signal_latex_expr}$")
        
        for expansion, num_terms in zip(self.expansions, self.num_terms):
            t = np.linspace(self.signal_bounds[0], self.signal_bounds[1], num_terms)
            ax.plot(t, expansion, label=f"{num_terms} Terms Expansion")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x(t)$")
        ax.set_title(rf"Fourier Series Expansion with Bounds {self.signal_bounds} and {len(self.t_val)} Samples")

    def plot_discretized_freq_domain(self, ax):
        ax.plot(self.freq, np.abs(self.fft_signal_freq), label="FFT frequency content")
         
        if self.discretized_fourier_transform is not None:
            has_begin = self.fourier_latex_expr.find("begin") != -1
            if has_begin:
                ax.plot(self.freq_val, self.discretized_fourier_transform, label=rf"{sp.pretty(self.fourier_transform.subs(self.w, sp.Symbol('ω')))}")
            else:
                # it's peicewise if the latex expr. contains begin, need to fall back to normal rendering
                ax.plot(self.freq_val, self.discretized_fourier_transform, label=rf"${self.fourier_latex_expr}$")
    
        ax.set_xlabel(rf"$\omega$")
        ax.set_ylabel(rf"$X(\omega)$")
        ax.set_title(rf"Discretized Fourier Transform with Bounds {self.fourier_bounds} and {len(self.freq_val)} Samples")

    def plot_error_of_fft_reconstruction(self, ax):
        ax.set_xlabel(rf"t")
        ax.set_ylabel(rf"Absolute Error")
        ax.set_title(rf"Absolute Error of Reconstructed Signal From FFT ")

        for expansion, num_terms in zip(self.expansions, self.num_terms):
            t = np.linspace(self.signal_bounds[0], self.signal_bounds[1], num_terms)
            error = abs(self.x_func(t) - expansion)
            ax.plot(t, error, label=f"{num_terms} Terms")
            
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

class DiracDelta(Signal):
    def __init__(self, t_substitution=None, signal_bounds=DEFAULT_SIGNAL_BOUNDS, signal_num_samples=DEFAULT_NUM_SIGNAL_SAMPLES,
                 fourier_bounds = DEFAULT_FOURIER_BOUNDS, fourier_num_samples=DEFAULT_NUM_FOURIER_SAMPLES):
        t = sp.symbols("t")
        delta = sp.DiracDelta(t)
        super().__init__(signal=delta, t_substitution=t_substitution,
                     signal_bounds=signal_bounds, signal_num_samples=signal_num_samples, 
                     fourier_bounds=fourier_bounds, fourier_num_samples=fourier_num_samples)
        
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

# class RectangularPulseTrain(Signal):
#     def __init__(self, t_substitution=None, signal_bounds=DEFAULT_SIGNAL_BOUNDS, signal_num_samples=DEFAULT_NUM_SIGNAL_SAMPLES,
#                  fourier_bounds = DEFAULT_FOURIER_BOUNDS, fourier_num_samples=DEFAULT_NUM_FOURIER_SAMPLES):
#         t, n = sp.symbols("t n")
#         u = sp.Heaviside(t)
#         p = u.subs(t, t + 0.5) - u.subs(t, t - 0.5)
#         N = 100 # Number of pulses
#         pulse_train = sp.Sum(p.subs(t, t - n), (n, -N, N))
#         pulse_train = pulse_train.doit()
#         super().__init__(signal=pulse_train, t_substitution=t_substitution,
#                      signal_bounds=signal_bounds, signal_num_samples=signal_num_samples, 
#                      fourier_bounds=fourier_bounds, fourier_num_samples=fourier_num_samples)

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

        # self.fourier_series_dict = self.compute_fourier_series()
        # self.coeffs = self.fourier_series_dict['coeffs']
        # self.freqs = self.fourier_series_dict['freqs']
        # periodicity_dict = self.check_periodicity(coefficients = self.coeffs, frequencies = self.freqs)
        # self.periodic = periodicity_dict['periodic']
        # self.expansions_term_numbers = [1, 3, 5, 10, 50, 100, 500, 1000]
        # self.expansions = [self.expand_fourier_series(self.coeffs, self.freqs, N) for N in self.expansions_term_numbers]

    # def compute_fourier_series(self):
    #     N = len(self.discretized_signal)
    #     coefficients = np.fft.fft(self.discretized_signal) / N
    #     a_0 = coefficients[0].real
    #     a_k = coefficients[1:N//2].real
    #     b_k = -coefficients[1:N//2].imag
    #     frequencies = np.fft.fftfreq(N)
    #     return {'coeffs': coefficients, 'freqs': frequencies / N, 'a_0': a_0, 'a_k': a_k, 'b_k': b_k}

    # def expand_fourier_series(self, coefficients, frequencies, N):
    #     reconstructed_signal = np.zeros_like(coefficients)
    #     for coeff, freq in zip(coefficients[:N], frequencies[:N]):
    #         reconstructed_signal += coeff * np.exp(2j * np.pi * freq * np.arange(len(coefficients)))
    #     return reconstructed_signal

    # def check_periodicity(self, coefficients, frequencies):
    #     # Compute magnitudes of the spectrum using coefficients
    #     magnitudes = np.abs(coefficients)
        
    #     # Find the highest peak index (excluding the DC component)
    #     highest_peak_index = np.argmax(magnitudes[1:]) + 1
        
    #     # Calculate the frequency corresponding to the highest peak
    #     frequency = frequencies[highest_peak_index]
        
    #     # Check if the frequency is close to 0 or 1
    #     is_periodic = np.isclose(frequency, 1.0) or np.isclose(frequency, 0.0)
        
    #     # Calculate the approximate period
    #     approximate_period = 1.0 / frequency
        
    #     return {
    #         "periodic": is_periodic,
    #         "approx_f0": frequency,
    #         "approx_T0": approximate_period
    #     }