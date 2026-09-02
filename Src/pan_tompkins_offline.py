import numpy as np

class PanTompkinsOffline:
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

    def _adaptive_threshold_offline(self, signal, peaks, peaks_indices, fs:float):
        """
        Classify detected ECG peaks using adaptive signal and noise peak estimates.

        This method is generalizable to both moving-window integrated ECG signals
        and band-pass-filtered ECG signals. It initializes signal and noise peak
        estimates from the first two seconds of the input signal, updates these
        estimates using detected local peaks, and computes primary and secondary
        adaptive thresholds. The detected peaks are then classified as signal or
        noise peaks according to the primary threshold.

        Parameters
        ----------
        signal : array_like
            Input ECG signal used to initialize the adaptive signal and noise
            peak estimates. This may be a moving-window integrated signal or
            a band-pass-filtered ECG signal.
        peaks : array_like
            Amplitudes of the detected local peaks in `signal`.
        peaks_indices : array_like
            Sample indices corresponding to `peaks`.
        fs : float
            Sampling frequency of `signal` in Hz.

        Returns
        -------
        threshold_1 : float
            Primary adaptive threshold calculated from the final signal and
            noise peak estimates.
        threshold_2 : float
            Secondary adaptive threshold, calculated as half of `threshold_1`.
        spk : float
            Final signal peak estimate.
        npk : float
            Final noise peak estimate.
        classified_sig_peaks : list
            Peaks classified as signal peaks.
        classified_noise_peaks : list
            Peaks classified as noise peaks.
        classified_sig_pks_indices : list
            Sample indices corresponding to the classified signal peaks.
        classified_noise_pks_indices : list
            Sample indices corresponding to the classified noise peaks.

        Notes
        -----
        The adaptive signal and noise peak estimates use the weighting scheme
        associated with the Pan-Tompkins QRS detection algorithm. The method
        provides an adaptive peak-classification stage and does not implement
        the complete Pan-Tompkins decision logic, including search-back and
        subsequent R-peak localization.

        The estimates and thresholds are updated according to:

            SPK[n] = 0.125 PEAK[n] + 0.875 SPK[n - 1]

            NPK[n] = 0.125 PEAK[n] + 0.875 NPK[n - 1]

            THRESHOLD_1 = NPK + 0.25(SPK - NPK)

            THRESHOLD_2 = 0.5 THRESHOLD_1

        See Also
        --------
        _detect_peaks
            Detects local peaks in the input ECG signal.
        _refractory_period
            Applies a minimum separation between accepted peak candidates.
        """

        # First 2 seconds of the `integrated signal` or `filtered signal`
        initial_signal = signal[:int(2 * fs)]

        # Initialize signal peak estimate (SPK) --> SPK0​ = 0.25 Xmax(Xint​)
        SPK = 0.25 * max(initial_signal)
        # Intialize noise peak estimate (NPK) --> NPK0 ​= 0.5 Xmean(Xint​)
        NPK = 0.5 * np.mean(initial_signal) 
        # Initialize threshold 1
        THRESHOLD_1 = NPK + 0.25 * (SPK - NPK)

        # Learning stage
        for peak in peaks:
            # Compare current peak with threshold
            if peak > THRESHOLD_1:
                # update SPK
                SPK = 0.125 * peak + 0.875 * SPK
            else:
                # update NPK
                NPK = 0.125 * peak + 0.875 * NPK

            # Update threshold 1
            THRESHOLD_1 = NPK + 0.25 * (SPK - NPK)

        # Calculate threshold 2
        THRESHOLD_2 = 0.5 * THRESHOLD_1

        # Initialize classiffied signal peaks & niose peaks and their indices
        classified_sig_peaks, classified_noise_peaks = ([] for i in range(2))
        classified_sig_pks_indices, classified_noise_pks_indices = ([] for i in range(2))

        # Classify signal peaks & noise peaks
        for i in range(len(peaks)):
            if peaks[i] > THRESHOLD_1:
                classified_sig_peaks.append(peaks[i])
                classified_sig_pks_indices.append(peaks_indices[i])
            else:
                classified_noise_peaks.append(peaks[i])
                classified_noise_pks_indices.append(peaks_indices[i])

        
        return (
            THRESHOLD_1, THRESHOLD_2, SPK, NPK,
            classified_sig_peaks, classified_noise_peaks,
            classified_sig_pks_indices, classified_noise_pks_indices
        )

    def _search_back(
        self, accepted_peaks, accepted_indices,
        noise_peaks, noise_indices, threshold_2
    ):
        """
        Recover potentially missed QRS complexes from rejected peak candidates
        when an unusually long RR interval is detected.

        The method examines consecutive accepted R-peak detections and compares
        each current RR interval with the average RR interval. If an RR interval
        exceeds 1.66 times the current average RR interval, the corresponding
        interval is considered suspicious. Rejected peak candidates within that
        interval are then examined, and candidates exceeding the secondary
        adaptive threshold are considered for recovery. If multiple eligible
        candidates exist, the candidate with the greatest amplitude is selected
        and inserted into the accepted detections.

        Parameters
        ----------
        accepted_peaks : array_like
            Amplitudes of the currently accepted R-peak candidates.
        accepted_indices : array_like
            Sample indices corresponding to `accepted_peaks`. The indices must
            be ordered chronologically.
        noise_peaks : array_like
            Amplitudes of peak candidates previously classified as noise.
        noise_indices : array_like
            Sample indices corresponding to `noise_peaks`.
        threshold_2 : float
            Secondary adaptive threshold used to determine whether a rejected
            peak is sufficiently strong to be reconsidered as a QRS candidate.

        Returns
        -------
        updated_peaks : list
            Accepted R-peak amplitudes after search-back recovery.
        updated_indices : list
            Sample indices corresponding to `updated_peaks`, ordered
            chronologically.

        Notes
        -----
        An RR interval is considered suspicious when:

            RR_current > 1.66 * RR_average

        For a suspicious interval, rejected candidates located between the two
        surrounding accepted R-peaks are examined. Only candidates exceeding
        `threshold_2` are considered, and the strongest eligible candidate is
        recovered.
        """

        # Copy accepted detections
        updated_peaks = list(accepted_peaks)
        updated_indices = list(accepted_indices)

        # Calculate initial RR intervals
        rr_intervals = [
            updated_indices[i] - updated_indices[i - 1]
            for i in range(1, len(updated_indices))
        ]

        # Need at least two RR intervals
        if len(rr_intervals) < 2:
            return updated_peaks, updated_indices

        # Initial average RR interval
        rr_average = np.mean(rr_intervals)

        i = 0

        while i < len(updated_indices) - 1:

            # Current RR interval
            rr_current = updated_indices[i + 1] - updated_indices[i]

            # Check for unusually long RR interval
            if rr_current > 1.66 * rr_average:

                lower_bound = updated_indices[i]
                upper_bound = updated_indices[i + 1]

                # Find eligible rejected candidates inside the interval
                candidates = [
                    (peak, index)
                    for peak, index in zip(noise_peaks, noise_indices)
                    if lower_bound < index < upper_bound
                    and peak > threshold_2
                ]

                if candidates:

                    # Select strongest candidate
                    recovered_peak, recovered_index = max(
                        candidates,
                        key=lambda candidate: candidate[0]
                    )

                    # Insert recovered QRS chronologically
                    updated_peaks.insert(i + 1, recovered_peak)
                    updated_indices.insert(i + 1, recovered_index)

                    # Recalculate RR intervals
                    rr_intervals = [
                        updated_indices[j] - updated_indices[j - 1]
                        for j in range(1, len(updated_indices))
                    ]

                    rr_average = np.mean(rr_intervals)

            i += 1

        return updated_peaks, updated_indices

    def detect(self, signal, fs, resample_fs=None):
        """
        Run the complete offline Pan-Tompkins ECG processing pipeline.

        Parameters
        ----------
        signal : array_like
            Raw ECG signal.
        fs : float
            Sampling frequency of the input ECG signal in Hz.
        resample_fs : float or None, optional
            Target sampling frequency for resampling. If None, the original
            sampling frequency is retained.

        Returns
        -------
        r_peaks : list
            Detected R-peak amplitudes.
        r_peak_indices : list
            Sample indices corresponding to the detected R-peaks.
        """
        
        # ---------------------------------------------------------
        # 1. Resampling
        # ---------------------------------------------------------
        if resample_fs is not None:
            signal = self.resample_signal(signal, fs, resample_fs)
            fs = resample_fs

        # ---------------------------------------------------------
        # 2. Band-pass filtering
        # ---------------------------------------------------------
        filtered_signal = self._bandpass_filter(signal, fs)

        # ---------------------------------------------------------
        # 3. Derivative
        # ---------------------------------------------------------
        derivative_signal = self._derivative(filtered_signal, fs)

        # ---------------------------------------------------------
        # 4. Squaring
        # ---------------------------------------------------------
        squared_signal = self._square(derivative_signal)

        # ---------------------------------------------------------
        # 5. Moving-window integration
        # ---------------------------------------------------------
        integrated_signal = self._moving_window_integration(
            squared_signal,
            fs
        )

        # ---------------------------------------------------------
        # 6. Detect local peaks in integrated signal
        # ---------------------------------------------------------
        integrated_peaks, integrated_indices = self._detect_peaks(
            integrated_signal
        )

        # ---------------------------------------------------------
        # 7. Adaptive thresholding — integrated domain
        # ---------------------------------------------------------
        (
            threshold_i1,
            threshold_i2,
            spk_i,
            npk_i,
            signal_peaks_i,
            noise_peaks_i,
            signal_indices_i,
            noise_indices_i
        ) = self._adaptive_threshold(
            integrated_signal,
            integrated_peaks,
            integrated_indices,
            fs
        )

        # ---------------------------------------------------------
        # 8. Refractory period — integrated domain
        # ---------------------------------------------------------
        accepted_peaks_i, accepted_indices_i = self._refractory_period(
            signal_peaks_i,
            signal_indices_i,
            fs
        )

        # ---------------------------------------------------------
        # 9. Detect local peaks in filtered ECG
        # ---------------------------------------------------------
        filtered_peaks, filtered_indices = self._detect_peaks(
            filtered_signal
        )

        # ---------------------------------------------------------
        # 10. Adaptive thresholding — filtered domain
        # ---------------------------------------------------------
        (
            threshold_f1,
            threshold_f2,
            spk_f,
            npk_f,
            signal_peaks_f,
            noise_peaks_f,
            signal_indices_f,
            noise_indices_f
        ) = self._adaptive_threshold(
            filtered_signal,
            filtered_peaks,
            filtered_indices,
            fs
        )

        # ---------------------------------------------------------
        # 11. Refractory period — filtered domain
        # ---------------------------------------------------------
        accepted_peaks_f, accepted_indices_f = self._refractory_period(
            signal_peaks_f,
            signal_indices_f,
            fs
        )

        # ---------------------------------------------------------
        # 12. R-peak localization
        # ---------------------------------------------------------
        localized_r_peaks, localized_r_indices = self._localize_r_peaks(
            filtered_signal,
            accepted_indices_i,
            fs
        )

        # ---------------------------------------------------------
        # 13. Search-back
        # ---------------------------------------------------------
        final_r_peaks, final_r_indices = self._search_back(
            localized_r_peaks,
            localized_r_indices,
            noise_peaks_f,
            noise_indices_f,
            threshold_f2
        )

        return final_r_peaks, final_r_indices
    

    