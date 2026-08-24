import numpy as np

class PanTompkinsPreprocessor:
    """Pan-Tompkins ECG QRS Preprocessing pipeline."""

    def __init__(self):
        pass

    @staticmethod
    def resample_signal(signal, original_fs:float, target_fs:float=200):
        """
        Resample a discrete signal to a target sampling frequency using
        linear interpolation.

        Parameters
        ----------
        signal : array_like
            Input discrete-time signal.
        original_fs : float
            Sampling frequency of the input signal in Hz.
        target_fs : float, optional
            Desired sampling frequency in Hz. Default is 200 Hz.

        Returns
        -------
        resampled_signal : ndarray
            Signal resampled at the target sampling frequency.
        new_time : ndarray
            Time vector corresponding to the resampled signal, in seconds.
        """

        # Calculate original time vector
        original_time = np.arange(len(signal)) / original_fs

        # Calculate new length = new number of samples
        new_length = int(len(signal) * target_fs / original_fs )

        # Calculate new time vector considering the new length and target sampling frequency
        new_time = np.arange(new_length) / target_fs

        # Apply linear interpolation
        resampled_signal = np.interp(new_time, original_time, signal)

        return resampled_signal, new_time

    def _band_pass_filter(self, signal):
        """
        Apply the recursive band-pass filter used in the Pan-Tompkins algorithm.

        The filter consists of a low-pass stage followed by a high-pass stage.
        Both stages are implemented using the recursive difference equations
        defined by the original Pan-Tompkins method.

        Parameters
        ----------
        signal : numpy.ndarray
            One-dimensional ECG signal to be filtered.

        Returns
        -------
        numpy.ndarray
            Band-pass filtered ECG signal.

        Notes
        -----
        The filter is implemented directly from its recursive difference
        equations rather than using a general-purpose digital filter design
        routine. The output is not amplitude-normalized.

        Band of frequencies: 5 - 15 Hz

        Mathematical statements: 

            Low-Pass: 
                y[n]= 2y[n−1] − y[n−2] + x[n] − 2x[n−6] + x[n−12]

            High-Pass:
                y[n]= 32x[n−16] − y[n−1] − x[n] + x[n−32]
        """

        # Low-pass stage
        # -----------------
        # Initialize low pass output --> also: [0.0] * len(signal)
        y_lp = np.zeros(len(signal))

        for sample in range(len(signal)):
            # Initialize y[n] = x[n]
            y_lp[sample] = signal[sample]

            # Apply 2y[n-1]
            if (sample >= 1):
                y_lp[sample] += 2 * y_lp[sample - 1]

            # Apply -y[n-2]
            if (sample >= 2):
                y_lp[sample] -= y_lp[sample - 2]

            # Apply -2x[n-6]
            if (sample >= 6):
                y_lp[sample] -= 2 * signal[sample - 6]

            # Apply x[n-12]
            if (sample >= 12):
                y_lp[sample] += signal[sample - 12]

        # High-pass stage
        # -----------------
        # Initialize high pass output
        y_hp = np.zeros(len(signal))
        
        for sample in range(len(signal)):
            # Initialize y[n] = -x[n]
            y_hp[sample] = -y_lp[sample] 

            # Apply 32x[n−16]
            if (sample >= 16):
                y_hp[sample] += 32 * y_lp[sample - 16]

            # Apply −y[n−1]
            if (sample >= 1):
                y_hp[sample] -= y_hp[sample - 1]

            # Apply x[n−32]
            if (sample >= 32):
                y_hp[sample] += y_lp[sample - 32]

        return y_hp


    def _derivative(self, signal, fs:float):
        """
        Apply the five-point derivative filter used in the Pan-Tompkins
        QRS detection algorithm.

        Parameters
        ----------
        signal : array_like
            Input ECG signal.
        fs : float
            Sampling frequency of the input signal in Hz.

        Returns
        -------
        derivative_signal : ndarray
            Derivative-filtered ECG signal.

        Notes
        -----
        The derivative is computed using the five-point finite-difference
        equation:

            y[n] = (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2]) / (8T)

        where T = 1 / fs is the sampling period.

        The first and last two samples cannot be evaluated using the
        five-point formulation and are therefore left at zero.
        """
        
        # Compute T = 1 / Fs
        T = 1 / fs
        
        # Initialize output
        derivative_signal = np.zeros(len(signal))

        # Apply 5-point derivative 
        for sample in range(2, len(signal) - 2):

            derivative_signal[sample] = (
                - signal[sample - 2]
                - 2 * signal[sample - 1]
                + 2 * signal[sample + 1]
                + signal[sample + 2]
            )

            derivative_signal[sample] /= 8 * T

        return derivative_signal

    def _square(self, signal):
        """
        Square each sample of a discrete signal.

        Parameters
        ----------
        signal : array_like
            Input signal.

        Returns
        -------
        squared_signal : ndarray
            Signal after squaring each sample.
        """

        # Initialize output
        squared_signal = np.zeros(len(signal))

        # Apply squaring
        for sample in range(len(signal)):
            squared_signal[sample] = signal[sample] ** 2

        return squared_signal

    def _moving_window_integration(self, signal, fs:float, window_duration:float=0.15):
        """
        Apply a moving-window integration to a discrete signal.

        Parameters
        ----------
        signal : array_like
            Input signal, typically the squared output of the
            Pan-Tompkins derivative stage.
        fs : float
            Sampling frequency of the input signal in Hz.
        window_duration : float, optional
            Integration-window duration in seconds. Default is 0.15 s.

        Returns
        -------
        integrated_signal : ndarray
            Moving-window integrated signal.

        Notes
        -----
        The moving-window integration is computed as:

            y[n] = (1/N) * sum(x[n-i], i=0,...,N-1)

        where N is the number of samples in the integration window.
        Samples before the first complete window are left at zero.
        """

        # Calculate window size
        window_size = int(window_duration * fs)

        # Initialize output
        integrated_signal = np.zeros(len(signal))

        # Calculate the first complete window
        window_sum = np.sum(signal[:window_size])                       # Sum of n samples in the first window
        integrated_signal[window_size - 1] = window_sum / window_size   # Mean of n samples in the first window

        # Move (slide) the window through the signal
        for sample in range(window_size, len(signal)):
            window_sum += signal[sample]                                # Add the next sample
            window_sum -= signal[sample - window_size]                  # Remove the previous sample (The sample that leaves the window)

            integrated_signal[sample] = window_sum / window_size        # Mean of the new window

        return integrated_signal

    def _detect_peaks(self, signal):
        """
        Detect local maxima in a one-dimensional signal.

        A sample is identified as a peak when its amplitude is greater than
        the preceding sample and greater than or equal to the following sample.

        Parameters
        ----------
        signal : array_like
            One-dimensional input signal in which local maxima are detected.

        Returns
        -------
        peaks : list
            Amplitudes of the detected local maxima.
        peaks_time : list
            Sample indices corresponding to the detected local maxima.

        Notes
        -----
        This method performs basic local-maximum detection without applying
        amplitude thresholds, a refractory period, or adaptive peak selection.
        These operations are handled by subsequent stages of the Pan-Tompkins
        algorithm.

        Mathematical statement: 
            x[n] > x[n−1] & x[n] ≥ x[n+1]
        """
        
        # Initialize lists to store peaks values 
        #   and their corresponding indices
        peaks = []
        peaks_indices = []

        # Apply the mathematical statement
        for sample in range(1, len(signal) - 1):
            if (
            signal[sample] > signal[sample - 1]
            and 
            signal[sample] >= signal[sample + 1]
            ):
                peaks.append(signal[sample]) 
                peaks_indices.append(sample)


        return peaks, peaks_indices

    def _refractory_period(self, peaks, peak_indices, fs:float, refractory_time:float=0.2):
        """
        Apply a refractory period to candidate peaks.

        Peaks occurring within the minimum allowed interval after an
        accepted peak are treated as belonging to the same cardiac event.
        When multiple candidates fall within this interval, only the
        candidate with the largest amplitude is retained.

        Parameters
        ----------
        peaks : array-like
            Amplitudes of the detected candidate peaks.
        peak_indices : array-like
            Sample indices corresponding to the detected candidate peaks.
        fs : float
            Sampling frequency of the signal in Hz.
        refractory_time : float, optional
            Minimum allowed time interval between accepted peaks, in seconds.
            Default is 0.2 seconds.

        Returns
        -------
        accepted_peaks : list
            Amplitudes of the peaks retained after applying the refractory period.
        accepted_indices : list
            Sample indices corresponding to the accepted peaks.
        """

        # Calculate number of refractory samples 
        #   = Minimum allowed distance between accepted peaks
        N = int(refractory_time * fs)

        # Initialize accepted peaks and accepted indices lists
        accepted_peaks = []
        accepted_indices = []

        # Consider the first peak candidate as accepted
        accepted_peaks.append(peaks[0])
        accepted_indices.append(peak_indices[0])

        # Go through the local candidate peaks
        for i in range(1, len(peaks)):

            # Check peaks distance and compare to 
            #   minimum allowed distance between accepted peaks
            if (peak_indices[i] - accepted_indices[-1]) >= N:

                # If true, one peak is outside the other's refractory period
                accepted_peaks.append(peaks[i])
                accepted_indices.append(peak_indices[i])

            else:
                # If false, the two local peaks are 
                #   inside the same refractory period so 
                #   check their amplitude to select the strongest
                if peaks[i] > accepted_peaks[-1]:
                    accepted_peaks[-1] = peaks[i]
                    accepted_indices[-1] = peak_indices[i]
                else:
                    continue

        return accepted_peaks, accepted_indices

    def _localize_r_peaks(self, signal, integrated_indices, fs:float, search_window:None=None):
        """
        Localize R-peaks in an ECG signal around candidate QRS locations.

        For each candidate index obtained from the moving-window integrated
        signal, searches a symmetric region of the input ECG signal and
        identifies the sample with the maximum amplitude as the corresponding
        R-peak. If `search_window` is not specified, a default half-window of
        150 ms is used.

        Parameters
        ----------
        signal : array_like
            ECG signal in which the R-peaks are to be localized.
        integrated_indices : array_like
            Sample indices of candidate QRS peaks detected from the
            moving-window integrated signal.
        fs : float
            Sampling frequency of the ECG signal in Hz.
        search_window : float or None, optional
            Search-window half-width in samples. If None, the half-width is
            set to 0.15 * fs, corresponding to 150 ms.

        Returns
        -------
        localized_r_peaks : list
            Amplitudes of the localized R-peaks.
        localized_r_peaks_indices : list
            Sample indices corresponding to the localized R-peaks.

        Notes
        -----
        The search is performed around each candidate index using the range

            [candidate_index - search_window,
            candidate_index + search_window]

        The sample with the maximum amplitude within each search region is
        selected as the localized R-peak.
        """

        # Check search window for any specific value
        if search_window == None:
            # Calculate search window
            search_window = int(0.15 * fs)

        # initialize a list to store localized r peaks 
        #   & it correspoding indices
        localized_r_peaks = []
        localized_r_peaks_indices = []

        for i in integrated_indices:
            # Calculate lower- and upper- bounds
            lowerbound = max(0, int(i - search_window))
            upperbound = min(len(signal), int(i + search_window))

            # Get the maximum amplitude in the area
            area = signal[lowerbound:upperbound]

            # Get indices
            local_index = np.argmax(area)
            global_index = lowerbound + local_index

            # collect r peaks
            localized_r_peaks.append(signal[global_index])

            # Collect r peaks indices
            localized_r_peaks_indices.append(global_index)

        return localized_r_peaks, localized_r_peaks_indices


