from abc import ABC
from abc import abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pypulseq.check_timing import __div_check as div_check
from pypulseq.opts import Opts


class EventBaseClass(ABC):
    """PyPulseq event base class.

    Attributes
    ----------
    delay : float
        Delay time in seconds (s).
    duration : float
        Total duration of the event in seconds (s).
    num_samples : int
        Number of samples of the event.
    raster_time : float
        Raster time in seconds (s).

    Methods
    -------
    plot()
        Create a plot of the event.
    """

    __slots__ = ["_delay", "_duration", "_num_samples", "_raster_time"]

    def __init__(self) -> None:
        """Initialize the EventBaseClass."""
        self._delay: float
        self._duration: float
        self._num_samples: int
        self._raster_time: float

    @property
    def delay(self) -> float:
        """Getter for delay."""
        return self._delay

    @delay.setter
    def delay(self, value: float):
        """Setter for delay.

        The setter ensures that the value is a positive number and a multiple
        of raster time and calls the private method _calc_duration() to
        update the duration of the event.

        Parameters
        ----------
        value : float
            delay of the event in seconds

        Raises
        ------
        ValueError
            If the delay is negative.
        ValueError
            If the delay is not a multiple of the raster time.
        """
        if value < 0:
            raise ValueError("The delay must be positive.")
        if not div_check(value, self._raster_time):
            raise ValueError("The delay must be a multiple of the raster time.")
        self._delay = value
        self._calc_duration()

    @property
    def duration(self) -> float:
        """Getter for duration."""
        return self._duration

    @property
    def num_samples(self) -> int:
        """Getter for num_samples."""
        return self._num_samples

    @property
    def raster_time(self) -> float:
        """Getter for raster_time."""
        return self._raster_time

    @abstractmethod
    def _calc_duration(self):
        """Calculate the total duration of the event.

        Will be implemented in the subclasses.
        """
        pass

    @abstractmethod
    def plot(self):
        """Create a plot of the event.

        Will be implemented in the subclasses.
        """
        pass


class Delay(EventBaseClass):
    """Delay event class."""

    __slots__ = []

    def __init__(self, delay: float, system: Opts = Opts()) -> None:
        """Initialize a delay event.

        Parameters
        ----------
        delay : float
            Duration of the delay in seconds
        system : Opts, optional
            System object containing system limits, by default Opts()
        """
        super().__init__()
        self._raster_time = system.rf_raster_time
        self._num_samples = 1
        self.delay = delay

    def __str__(self) -> str:
        info = f"""Delay event:
           Delay: {self._delay} s
           Duration: {self._duration} s
           Raster time: {self._raster_time} s
           Number of samples: {self._num_samples}\n"""
        return info

    @EventBaseClass.num_samples.setter
    def num_samples(self, value) -> None:
        print("Number of samples cannot be set for Delay event.")

    def _calc_duration(self) -> float:
        """Calculate the duration of the delay event.

        This method calculates the duration of the delay event. For a delay event the total duration is equal to the '_delay' attribute.

        Returns
        -------
        float
            Total duration of the delay event.
        """
        self._duration = self._delay
        return self._duration

    def plot(self) -> None:
        print("Plot function not implemented for Delay class.")


class ADC(EventBaseClass):
    """ADC event class."""

    __slots__ = ["_dwell", "_freq_offset", "_phase_offset"]

    def __init__(
        self,
        num_samples: int,
        delay: float = 0,
        dwell: float = None,
        duration: float = None,
        freq_offset: float = 0,
        phase_offset: float = 0,
        system: Opts = Opts(),
    ) -> None:
        """Initialize an ADC event.

        Parameters
        ----------
        num_samples : int
            Number of samples of the ADC event.
        delay : float, optional
            Delay of the ADC event in seconds (s), by default 0
        dwell : float, optional
            Dwell time of the ADC event in seconds (s), by default None
        duration : float, optional
            Acquisition time of the ADC event in seconds (s), by default None
        freq_offset : float, optional
            Frequency offset of the ADC event in Hertz (Hz), by default 0
        phase_offset : float, optional
            Phase offset of the ADC event in radians, by default 0
        system : Opts, optional
            System object containing system limits, by default Opts()

        Raises
        ------
        ValueError
            If neither dwell nor duration is set.
        ValueError
            If both dwell and duration are set.
        ValueError
            If the number of samples is not positive.
        ValueError
            If the duration is negative.
        ValueError
            If the dwell is negative.
        """

        super().__init__()

        if dwell is None and duration is None:
            raise ValueError("Either dwell or duration must be set.")
        if dwell is not None and duration is not None:
            raise ValueError("Only one of dwell or duration can be set.")
        if num_samples < 1:
            raise ValueError("Number of samples must be positive.")
        if duration is not None and duration < 0:
            raise ValueError("Duration must be positive.")
        if dwell is not None and dwell < 0:
            raise ValueError("Dwell must be positive.")
        if delay < 0:
            raise ValueError("Delay must be positive.")

        self._raster_time = system.adc_raster_time
        self._freq_offset = freq_offset
        self._phase_offset = phase_offset
        self._num_samples = num_samples
        self._delay = delay

        if duration is not None and duration > 0:
            self.dwell = duration / num_samples

        if dwell is not None and dwell > 0:
            # ToDo: check if dwell is a multiple of the raster time?!
            self._dwell = dwell
            duration = dwell * num_samples
            self._calc_duration()

    @property
    def dwell(self) -> float:
        """Getter for dwell."""
        return self._dwell

    @property
    def freq_offset(self) -> float:
        """Getter for freq_offset."""
        return self._freq_offset

    @freq_offset.setter
    def freq_offset(self, value: float):
        """Setter for freq_offset."""
        self._freq_offset = value

    @property
    def phase_offset(self) -> float:
        """Getter for phase_offset."""
        return self._phase_offset

    @phase_offset.setter
    def phase_offset(self, value: float):
        """Setter for phase_offset."""
        self._phase_offset = value

    def _calc_duration(self) -> float:
        """Caculate the duration of the ADC event.

        This method calculates the duration of the ADC event based on the number
        of samples, the dwell time and the ADC delay time.

        Returns
        -------
        float
            Total duration of the ADC event in seconds.

        Raises
        ------
        ValueError
            If the total duration of the ADC event is negative.
        """
        _total_dur = self._num_samples * self._dwell + self._delay
        if not _total_dur >= 0:
            raise ValueError("The total duration of the ADC event must be greater than or equal to zero.")
        self._duration = _total_dur
        return self._duration

    def plot(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(np.ones(self._num_samples) * self.raster_time, np.zeros(self._num_samples), "x", "red")


# TODO: Not complete yet. Need to add the attributes and methods.
class RfBase(EventBaseClass):
    """RF event base class."""

    def __init__(self) -> None:
        super().__init__()


# TODO: Not complete yet. Need to add the attributes and methods.
class RfGauss(RfBase):
    """Gaussian RF pulse event."""

    def __init__(self) -> None:
        super().__init__()


# TODO: Not complete yet. Need to add the attributes and methods.
class RfSinc(RfBase):
    """Sinc RF pulse event."""

    def __init__(self) -> None:
        super().__init__()


# TODO: Not complete yet. Need to add the attributes and methods.
class RfBlock(RfBase):
    """Block RF pulse event."""

    def __init__(self) -> None:
        super().__init__()


# TODO: Not complete yet. Need to add the attributes and methods.
class RfArbitrary(RfBase):
    """Arbitrary RF pulse event."""

    def __init__(self) -> None:
        super().__init__()


# TODO: Not complete yet. Need to add the attributes and methods.
class GradientBase(EventBaseClass):
    """Gradient event base class."""

    def __init__(self) -> None:
        super().__init__()


# TODO: Not complete yet. Need to add the attributes and methods.
class GradTrap(GradientBase):
    """Trapezoidal gradient event."""

    def __init__(self) -> None:
        super().__init__()


# TODO: Not complete yet. Need to add the attributes and methods.
class GradArbitrary(GradientBase):
    """Arbitrary gradient event."""

    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    sys = Opts(
        adc_raster_time=100e-9,
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
        max_grad=30,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
    )

    d1 = Delay(delay=1.0, system=sys)
    print(d1.duration)

    a1 = ADC(num_samples=256, delay=1.0, dwell=1e-3, system=sys)

    pass
