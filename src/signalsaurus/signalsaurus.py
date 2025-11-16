from collections.abc import Callable
import math
import copy
from functools import cached_property
from typing import Self, override

import numpy as np
import scipy.signal as signal


class Spectrum:
    def __init__(
        self,
        f_sample: float,
        df: float,
        xs: None | Callable[[np.ndarray[float]], np.ndarray[float]] = None,
    ) -> None:
        self.f_sample = f_sample
        self.df = df
        if xs is None:
            self._xs = None
        else:
            self.xs = xs(self.ts)

    @override
    def __str__(self) -> str:
        try:
            self.xs
            return f"Spectrum(f_sample={self.f_sample}, df={self.df}, N_samples={self.N_samples}, xs set)"
        except AttributeError:
            return f"Spectrum(f_sample={self.f_sample}, df={self.df}, N_samples={self.N_samples}, xs NOT set)"

    @override
    def __repr__(self) -> str:
        return str(self)

    @property
    def f_sample(self) -> float:
        """Sampling rate in Hz"""
        return self._f_sample

    @f_sample.setter
    def f_sample(self, f_sample: float) -> None:
        try:
            del self.ts
        except AttributeError:
            pass
        self._f_sample = f_sample
        try:
            self.df = self.df
        except AttributeError:
            pass

    @property
    def df(self) -> float:
        """Frequency bin width in Hz"""
        return self._df

    @df.setter
    def df(self, df: float) -> None:
        try:
            del self.ts
        except AttributeError:
            pass
        try:
            del self.xs
        except AttributeError:
            pass
        try:
            del self.Xs
        except AttributeError:
            pass
        try:
            del self.Sxxs
        except AttributeError:
            pass
        N_samples = math.ceil(self.f_sample / df)
        try:
            if N_samples != self.N_samples:
                del self.xs
        except AttributeError:
            pass
        self._df = self.f_sample / N_samples

    @property
    def T_sample(self) -> float:
        """Sample period in seconds"""
        return 1 / self.f_sample

    @T_sample.setter
    def T_sample(self, T_sample: float) -> None:
        self.f_sample = 1 / T_sample

    @property
    def T_max(self) -> float:
        "Total experiment time in seconds"
        return 1 / self.df

    @T_max.setter
    def T_max(self, T_max: float) -> None:
        self.df = 1 / T_max

    @property
    def f_Ny(self) -> float:
        """Nyquist frequency in Hz"""
        return self.f_sample / 2.0

    @f_Ny.setter
    def f_Ny(self, f_Ny: float) -> None:
        self.f_sample = f_Ny * 2.0

    @property
    def N_samples(self) -> int:
        """Number of samples in spectrum"""
        return math.ceil(self.f_sample / self.df)

    @cached_property
    def ts(self) -> np.ndarray[float]:
        """Array of sample times in time domain"""
        return np.arange(self.N_samples, dtype=float) * self.T_sample

    @property
    def xs(self) -> np.ndarray[float]:
        """Time domain data array in Volts (presumably)"""
        return self._xs

    @xs.setter
    def xs(self, xs: np.ndarray[float]) -> None:
        if len(xs) != self.N_samples:
            raise ValueError(
                f"len(xs) ({len(xs)}) must match N_samples ({self.N_samples}))"
            )
        self._xs = xs
        try:
            del self.Xs
        except AttributeError:
            pass
        try:
            del self.Sxxs
        except AttributeError:
            pass

    def _set_xs(self, xs: np.ndarray[float]) -> None:
        self._xs = xs

    @cached_property
    def fs(self) -> np.ndarray[float]:
        """Array of frequency bins in frequency domain"""
        return np.fft.rfftfreq(self.N_samples, d=self.T_sample)

    @cached_property
    def Xs(self) -> np.ndarray[float]:
        """DFT of data"""
        try:
            return np.fft.rfft(self.xs)
        except IndexError as e:
            raise ValueError("Must specify xs before using Xs!") from e

    @cached_property
    def Sxxs(self) -> np.ndarray[float]:
        """Power spectral density in V^2/Hz"""
        try:
            return np.real(np.conj(self.Xs) * self.Xs) / (
                self.N_samples * self.f_sample
            )
        except IndexError as e:
            raise ValueError("Must specify xs before using Sxxs!") from e

    def deepcopy(self) -> Self:
        return copy.deepcopy(self)

    def apply_filter(
        self, b, a, inplace: bool = False, timedomain=False, reverse=False
    ) -> Self:
        """Apply a filter with coefficient arrays b (numerator) and a (denominator)

                b[0]*(jw)**M + b[1]*(jw)**(M-1) + ... + b[M]
        H(w) = ----------------------------------------------
                a[0]*(jw)**N + a[1]*(jw)**(N-1) + ... + a[N]

        If `inplace`, then replaces the existing the existing Spectrum and returns self

        If `timedomain`, performs the filtering in the time domain

        If `reverse`, applies the time-reversed filter (`xs` is ultimately in the
            forward sense regardless)
        """
        if timedomain:
            b_dig, a_dig = signal.bilinear(b, a, self.f_sample)
            zi = signal.lfilter_zi(b_dig, a_dig)
            xs = self.xs
            if reverse:
                xs = xs[::-1]
            if inplace:
                ys, _ = signal.lfilter(b_dig, a_dig, xs, zi=zi * xs[0])
                if reverse:
                    ys = ys[::-1]
                self.xs = ys
                return self
            else:
                spectrum = Spectrum(self.f_sample, self.df)
                ys, _ = signal.lfilter(b_dig, a_dig, xs, zi=zi * xs[0])
                if reverse:
                    ys = ys[::-1]
                spectrum.xs = ys
                return spectrum
        else:
            _, Hs = signal.freqs(b, a, self.fs * 2 * np.pi)
            if reverse:
                Hs = np.conj(Hs)
            if inplace:
                self.Xs *= Hs
                self._set_xs(np.fft.irfft(self.Xs))
                return self
            else:
                spectrum = Spectrum(self.f_sample, self.df)
                Xs = self.Xs * Hs
                xs = np.fft.irfft(Xs)
                spectrum.xs = xs
                spectrum.Xs = Xs
                return spectrum
