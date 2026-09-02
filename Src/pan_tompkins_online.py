import numpy as np

class PanTompkinsOnline:
    def __init__(self):
        # History / Memory

        self.input_history = []
        self.lp_history = []
        self.hp_history = []

        self.derivative_history = []
        self.y_derivative = []

        self.y_square_hist = []

        self.mwi_history = []
        self.y_integrated = []

        self.local_pks_history = []
        self.peaks_hist = []
        self.peaks_indices_hist = []

        self.refractory_period_history_pk = []
        self.refractory_period_history_id = []
        self.accepted_peaks = []
        self.accepted_indices = []

        self.initial_samples_history = []
        self.initial_samples_index_history = []
        self.spk_hist = []
        self.spk_index_hist = []
        self.npk_hist = []
        self.npk_index_hist = []
        self.thre1_hist = []
        self.thre1_index_hist = []
        self.thre2_hist = []
        self.thre2_index_hist = []
        self.y_adopted_peaks = []
        self.y_adopted_peaks_indices = []
        self.y_adopted_noises = []
        self.y_adopted_noises_indices = []

        self.filtered_history = []          
        self.filtered_index_history = []   

        self.search_back_peaks = []
        self.search_back_indices = []
        self.search_back_flags = [] 

        self.pending_candidates = []       
        self.localized_r_peak_hist = []
        self.localized_r_peak_index_hist = []

        # Running sum of the current MWI window
        self.window_sum = 0.0
        # Adoptive thresholding global variables
        self.initialization_complete = False
        self.SPK = None
        self.NPK = None
        self.THRESHOLD_1 = None
        self.THRESHOLD_2 = None

    def resample_signal(self, signal, original_fs:float, target_fs:float=200):
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

    def _band_pass_filter(self, sample:float):
        """
        Process one incoming ECG sample through the recursive band-pass
        filter used in the online Pan-Tompkins algorithm.

        The filter consists of a low-pass stage followed by a high-pass
        stage. Both stages are implemented using the recursive difference
        equations defined by the original Pan-Tompkins method. Unlike the
        offline implementation, this method processes one sample at a time
        and maintains the previous input and filter-output samples in
        persistent history buffers.

        Parameters
        ----------
        sample : float
            Current ECG sample to be processed.

        Returns
        -------
        y_hp : float
            Current high-pass filter output after low-pass filter, corresponding to the
            band-pass-filtered ECG sample.

        Notes
        -----
        The filter is implemented directly from the recursive difference
        equations rather than using a general-purpose digital filter design
        routine. The output is not amplitude-normalized.

        The filter stages are:

            Low-Pass:
                y_LP[n] = 2y_LP[n−1] − y_LP[n−2]
                        + x[n] − 2x[n−6] + x[n−12]

            High-Pass:
                y_HP[n] = 32y_LP[n−16] − y_HP[n−1]
                        − y_LP[n] + y_LP[n−32]

        The method is stateful: previous input samples and filter outputs
        are required to calculate the current output. Therefore, samples
        must be supplied sequentially.

        In the online implementation, the required previous samples are
        maintained in memory rather than recomputing the filter over the
        complete ECG signal for every new sample.

        The filter corresponds to the recursive band-pass filtering stage
        of the Pan-Tompkins QRS detection algorithm and is intended for
        sample-by-sample real-time processing.
        """

        # Store the new input sample
        self.input_history.append(sample)

        # Current sample index
        n = len(self.input_history) - 1

        # Initialize current low-pass output
        y_lp = self.input_history[n]

        # Apply 2y[n-1]
        if n >= 1:
            y_lp += 2 * self.lp_history[n - 1]

        # Apply -y[n-2]
        if n >= 2:
            y_lp -= self.lp_history[n - 2]

        # Apply -2x[n-6]
        if n >= 6:
            y_lp -= 2 * self.input_history[n - 6]

        # Apply x[n-12]
        if n >= 12:
            y_lp += self.input_history[n - 12]

        # Store the current low-pass output
        self.lp_history.append(y_lp)

        # Initialize y[n] = -x[n]
        y_hp = -1 * y_lp

        # Apply 32x[n−16]
        if n >= 16:
            y_hp += 32 * self.lp_history[n - 16]

        # Apply −y[n−1]
        if n >= 1:
            y_hp -= self.hp_history[n - 1]

        # Apply x[n−32]
        if n >= 32:
            y_hp += self.lp_history[n - 32]

        # Store current high-pass output
        self.hp_history.append(y_hp)

        return y_hp

    def _derivative(self, sample:float, fs:float):
        """
        Compute the derivative of one incoming ECG sample using the
        five-point finite-difference equation from the Pan-Tompkins algorithm.

        The function operates sequentially on one incoming sample at a time.
        Each sample is stored in an internal history buffer. Once at least
        five samples are available, the derivative of the center sample is
        calculated using the two preceding and two succeeding samples.
        Consequently, the computed derivative corresponds to a sample that
        is two samples behind the most recently received sample.

        Parameters
        ----------
        sample : float
            Current band-pass-filtered ECG sample.
        fs : float
            Sampling frequency of the input ECG signal in Hz.

        Returns
        -------
        float
            Computed derivative sample. During the initial warm-up period,
            when fewer than five samples are available, 0.0 is returned.

        Notes
        -----
        The derivative is computed using the five-point finite-difference
        equation:

            y[n] = (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2]) / (8T)

        where T = 1 / fs is the sampling period.

        Because the equation requires two future samples, the derivative
        output is delayed by two samples relative to the incoming signal.
        The first valid derivative is therefore obtained after five input
        samples have been received.

        The derivative history stores the incoming filtered samples and
        serves as the memory required for sequential online processing.
        """
        
        # Compute T = 1 / Fs
        T = 1 / fs

        # Store the sample
        self.derivative_history.append(sample)

        # Current sample index 
        n = len(self.derivative_history) - 1

        # Initialize current derivative output
        y_der = self.derivative_history[n] 

        # Apply 5-point derivative 
        if len(self.derivative_history) >= 5:
            # Samples center, e.g: [0, 1, 2, 3, 4] --> if n = 4 then center = 2
            center = n - 2

            # Calculate derivative of sample
            y_der = (
                - self.derivative_history[center - 2]
                - 2 * self.derivative_history[center - 1]
                + 2 * self.derivative_history[center + 1]
                + self.derivative_history[center + 2]
            )
            
            y_der /= 8 * T

            # Add current calculated output to the output list
            self.y_derivative.append(y_der)

        else:
            # Just in case of indexing error
            return 0.0
        
        return y_der

    def _square(self, sample:float):
        """
        Square one incoming ECG sample in the online Pan-Tompkins pipeline.

        Parameters
        ----------
        sample : float
            Current derivative sample.

        Returns
        -------
        float
            Squared value of the input sample.

        Notes
        -----
        The squaring operation is applied independently to each incoming
        derivative sample according to:

            y[n] = x[n]^2

        The squared output is stored in `y_square_hist` for inspection
        and visualization of the processed signal. The history is not
        required for the squaring operation itself.
        """

        # Apply squaring on the sample
        y_square = sample ** 2

        # Add the output to the outputs list
        self.y_square_hist.append(y_square)

        return y_square

    def _moving_window_integration(self, sample:float, fs:float, window_duration:float=0.15):
        """
        Compute the moving-window integration of one incoming ECG sample.

        The method implements the moving-window integration stage of the
        Pan-Tompkins QRS detection algorithm in an online manner. A fixed
        integration window is maintained over the most recent samples. Once
        the first complete window is available, its mean is calculated.
        For every subsequent sample, the running window sum is updated by
        adding the newest sample and removing the oldest sample, avoiding
        recalculation of the entire window.

        Parameters
        ----------
        sample : float
            Current incoming sample of the squared ECG signal.
        fs : float
            Sampling frequency of the input signal in Hz.
        window_duration : float, optional
            Duration of the moving integration window in seconds.
            Default is 0.15 seconds.

        Returns
        -------
        y_mwi : float or None
            Current moving-window integrated value. Returns ``None`` until
            the first complete integration window has been accumulated.

        Notes
        -----
        The window length is calculated as:

            N = int(window_duration * fs)

        For a complete window, the integrated output is:

            y[n] = (1/N) * sum(x[n-k]),  k = 0, ..., N-1

        During online processing, the running sum is updated recursively:

            S[n] = S[n-1] + x[n] - x[n-N]

        This avoids recalculating the sum of all samples in the window
        for every incoming sample.
        """

        # Calculate window size
        window_size = int(window_duration * fs)

        # Store the incoming sample
        self.mwi_history.append(sample)

        # Number of samples currently available
        n_samples = len(self.mwi_history)

        # Not enough samples for a complete window yet
        if n_samples < window_size:
            return None

        # First complete window
        if n_samples == window_size:
            # Sum of all samples in the first complete window
            self.window_sum = np.sum(self.mwi_history[:window_size])

        # Every subsequent sample: slide the window
        else:
            # Add the newest sample
            self.window_sum += self.mwi_history[-1]

            # Remove the oldest sample leaving the window
            self.window_sum -= self.mwi_history[-window_size - 1]

        # Calculate the moving average
        y_mwi = self.window_sum / window_size

        # Store output for inspection/visualization
        self.y_integrated.append(y_mwi)

        return y_mwi

    def _detect_peaks(self, sample:float):
        """
        Detect local maxima in the moving-window-integrated ECG signal.

        A sample is identified as a local peak when it is greater than the
        preceding sample and greater than or equal to the following sample.
        Because the following sample is required for the comparison, the
        candidate peak is evaluated with a one-sample delay.

        Parameters
        ----------
        sample : float
            Current incoming sample from the moving-window-integrated signal.

        Returns
        -------
        y_loc_pk : float or None
            Amplitude of the detected local peak. Returns None when no peak
            can be confirmed.
        y_loc_pk_index : int or None
            Sample index of the detected local peak. Returns None when no
            peak is detected.

        Notes
        -----
        This method performs basic local-maximum detection without applying
        amplitude thresholds, a refractory period, or adaptive peak selection.
        These operations are handled by subsequent stages of the Pan-Tompkins
        algorithm.

        Mathematical statement: 
            x[n] > x[n−1] & x[n] ≥ x[n+1]
        """

        # Handle missing samples
        if sample is None:
            sample = 0.0

        # Store the incoming sample
        self.local_pks_history.append(sample)

        # Initialize outputs
        y_loc_pk = None
        y_loc_pk_index = None

        # Current sample index
        n = len(self.local_pks_history) - 1

        # At least three samples are required
        if len(self.local_pks_history) >= 3:

            # The middle sample is the candidate
            center = n - 1

            # Check whether the candidate is a local maximum
            if (
                self.local_pks_history[center] > self.local_pks_history[center - 1]
                and
                self.local_pks_history[center] >= self.local_pks_history[center + 1]
            ):

                # Store detected peak
                y_loc_pk = self.local_pks_history[center]
                y_loc_pk_index = center

                # Store peak and index
                self.peaks_hist.append(y_loc_pk)
                self.peaks_indices_hist.append(y_loc_pk_index)
        else:
            return (None, None)

        return y_loc_pk, y_loc_pk_index

    def _refractory_period(self, peak:float, peak_index:int, fs:float, refractory_time:float=0.2):
        """
        Apply a refractory period to incoming ECG peak candidates.

        The refractory period prevents two closely spaced peak candidates
        from being accepted as separate QRS complexes. A candidate is
        accepted when its sample index is sufficiently distant from the
        most recently accepted candidate. If a new candidate occurs within
        the refractory period, its amplitude is compared with the most
        recently accepted candidate and the stronger candidate is retained.

        Parameters
        ----------
        peak : float
            Amplitude of the incoming peak candidate.
        peak_index : int
            Sample index of the incoming peak candidate.
        fs : float
            Sampling frequency of the ECG signal in Hz.
        refractory_time : float, optional
            Minimum allowed time between accepted peak candidates, in
            seconds. Default is 0.2 seconds (200 ms).

        Returns
        -------
        y_acc_peak : float or None
            Accepted peak amplitude. Returns ``None`` when the incoming
            candidate is rejected and does not replace the previous
            accepted candidate.
        y_acc_peak_index : int or None
            Sample index corresponding to ``y_acc_peak``. Returns ``None``
            when the incoming candidate is rejected.

        Notes
        -----
        The refractory period is converted from seconds to samples as:

            N = refractory_time * fs

        The online implementation processes one candidate at a time and
        therefore maintains the previously accepted peaks and their
        indices as persistent state.

        If a candidate occurs within the refractory period of the previous
        accepted candidate, the candidate with the greater amplitude is
        retained.
        """
        # Handle missing samples
        if peak is None or peak_index is None:
            return None, None

        # Calculate number of refractory samples 
        #   = Minimum allowed distance between accepted peaks
        N = int(refractory_time * fs)

        # Store the incoming peak and index
        self.refractory_period_history_pk.append(peak)
        self.refractory_period_history_id.append(peak_index)

        # Initialize outputs
        y_acc_peak = None
        y_acc_peak_index = None

        # Check for the first candidate
        if len(self.accepted_indices) == 0:

            # Assign peak and index
            y_acc_peak = peak
            y_acc_peak_index = peak_index

            # Store the first candidate
            self.accepted_peaks.append(peak)
            self.accepted_indices.append(peak_index)

        else:
            # Calculate distance from last accepted candidate 
            distance = (peak_index - self.accepted_indices[-1]) 

            # Compare distance with allowed refractory period
            if distance >= N: 

                # If true, one peak is outside the other's refractory period 
                y_acc_peak = peak
                y_acc_peak_index = peak_index

                # Store peak and index
                self.accepted_peaks.append(y_acc_peak)
                self.accepted_indices.append(y_acc_peak_index)

            else:
                # If false, the two local peaks are 
                #   inside the same refractory period so 
                #   check their amplitude to select the strongest
                if peak > self.accepted_peaks[-1]:

                    # Assign the strongest peak
                    y_acc_peak = peak
                    y_acc_peak_index = peak_index

                    # Store the strongest peak and its index
                    self.accepted_peaks[-1] = y_acc_peak
                    self.accepted_indices[-1] = y_acc_peak_index

        return y_acc_peak, y_acc_peak_index

    def _adoptive_threshold_online(
        self,
        sample: float,
        sample_index: int,
        peak: float | None,
        peak_index: int | None,
        fs: float
    ):
        """
        Perform the online adaptive thresholding stage of the
        Pan-Tompkins QRS detection algorithm.

        The function first performs a two-second initialization period
        to estimate the initial signal peak level (SPK), noise peak level
        (NPK), and adaptive thresholds. After initialization, each detected
        candidate peak is classified as either a signal peak or a noise
        peak using the primary adaptive threshold. The corresponding
        SPK or NPK estimate is then updated, followed by recalculation
        of the adaptive thresholds.

        Parameters
        ----------
        sample : float
            Current incoming sample from the signal stream. During the
            initialization period, these samples are collected to estimate
            the initial signal and noise peak levels.

        sample_index : int
            Index of the current incoming sample in the signal stream.

        peak : float or None
            Current local peak candidate detected from the moving-window
            integrated signal. If no candidate peak is available at the
            current step, this should be None.

        peak_index : int or None
            Sample index corresponding to the current candidate peak.

        fs : float
            Sampling frequency of the input signal in Hz.

        Returns
        -------
        classified_peak : float or None
            Candidate peak classified as a signal peak. Returns None when
            the candidate is classified as noise or when the algorithm is
            still in the initialization stage.

        classified_peak_index : int or None
            Sample index corresponding to the classified signal peak.
            Returns None when no signal peak is classified.

        classified_noise : float or None
            Candidate peak classified as a noise peak. Returns None when
            the candidate is classified as a signal peak or when the
            algorithm is still in the initialization stage.

        classified_noise_index : int or None
            Sample index corresponding to the classified noise peak.
            Returns None when no noise peak is classified.

        Notes
        -----
        The first two seconds of the incoming signal are used to initialize
        the adaptive signal and noise peak estimates:

            SPK = 0.25 * max(initial_samples)

            NPK = 0.5 * mean(initial_samples)

        The primary threshold is calculated as:

            THRESHOLD_1 = NPK + 0.25 * (SPK - NPK)

        The secondary threshold is calculated as:

            THRESHOLD_2 = 0.5 * THRESHOLD_1

        After initialization, each candidate peak is classified according
        to the primary threshold.

        For a signal peak:

            SPK[n] = 0.125 * PEAK[n] + 0.875 * SPK[n-1]

        For a noise peak:

            NPK[n] = 0.125 * PEAK[n] + 0.875 * NPK[n-1]

        After each classification, both adaptive thresholds are recalculated.

        The function maintains the adaptive estimates and threshold values
        through global state variables for the current notebook
        implementation. These will later be converted to instance state
        when the algorithm is integrated into the final class.
        """

        # Initialize the outputs
        classified_peak = None
        classified_peak_index = None
        classified_noise = None
        classified_noise_index = None

        # Initialization
        if not self.initialization_complete:

            # Check for sanples in the first 2 seconds
            if (sample_index < 2 * fs) and sample is not None:
                self.initial_samples_history.append(sample)
                self.initial_samples_index_history.append(sample_index)

                return 0.0, 0.0, 0.0, 0.0

            if len(self.initial_samples_history) != 0:
                # Initialize signal peak estimate (SPK) --> SPKI0​ = 0.25 Xmax(Xint​)
                self.SPK = 0.25 * max(self.initial_samples_history)
                self.spk_hist.append(self.SPK)
                self.spk_index_hist.append(sample_index)

                # Intialize noise peak estimate (NPK) --> NPK0 ​= 0.5 Xmean(Xint​)
                self.NPK = 0.5 * np.mean(self.initial_samples_history)
                self.npk_hist.append(self.NPK)
                self.npk_index_hist.append(sample_index)

                # Initialize threshold 1
                self.THRESHOLD_1 = self.NPK + 0.25 * (self.SPK - self.NPK)
                self.thre1_hist.append(self.THRESHOLD_1)
                self.thre1_index_hist.append(sample_index)

                # Initialize threshold 2
                self.THRESHOLD_2 = 0.5 * self.THRESHOLD_1
                self.thre2_hist.append(self.THRESHOLD_2)
                self.thre2_index_hist.append(sample_index)

                # Set initialization as complete
                self.initialization_complete = True

                return 0.0, 0.0, 0.0, 0.0

        # No candidate peak
        if peak is None or peak_index is None:
            return None, None, None, None

        # Learning stage / Candidate peak classification

        # Check for peak by comparing to current learned threshold
        if peak > self.THRESHOLD_1:

            # Update SPK
            self.SPK = 0.125 * peak + 0.875 * self.SPK

            # Store adopted peak and its index
            self.y_adopted_peaks.append(peak)
            self.y_adopted_peaks_indices.append(peak_index)

            # Assign the peak and its index
            classified_peak = peak
            classified_peak_index = peak_index

            # Store SPK state
            self.spk_hist.append(self.SPK)
            self.spk_index_hist.append(peak_index)

        else:

            # Update NPK
            self.NPK = 0.125 * peak + 0.875 * self.NPK

            # Store adopted noise and its index
            self.y_adopted_noises.append(peak)
            self.y_adopted_noises_indices.append(peak_index)

            # Assign the noise and its index
            classified_noise = peak
            classified_noise_index = peak_index

            # Store NPK state
            self.npk_hist.append(self.NPK)
            self.npk_index_hist.append(peak_index)

        # Update threshold 1
        self.THRESHOLD_1 = self.NPK + 0.25 * (self.SPK - self.NPK)
        self.thre1_hist.append(self.THRESHOLD_1)
        self.thre1_index_hist.append(peak_index)

        # Update threshold 2
        self.THRESHOLD_2 = 0.5 * self.THRESHOLD_1
        self.thre2_hist.append(self.THRESHOLD_2)
        self.thre2_index_hist.append(peak_index)

        return (
            classified_peak,
            classified_peak_index,
            classified_noise,
            classified_noise_index
        )

    def _localize_r_peak(
        self,
        sample_index: int,
        candidate_index: int | None,
        fs: float,
        search_window: int | None = None,
        mwi_delay: int | None = None
    ):
        """
        Localize the exact R-peak in the band-pass filtered ECG signal around a candidate
        detected from the moving-window integrated signal.

        The function works fully online. It assumes that every filtered sample has already
        been stored in the global lists ``filtered_history`` and ``filtered_index_history``
        (this is done in the main processing loop right after the band-pass filter).
        Candidates are queued until their complete search window is available, then the
        strongest peak (absolute maximum) inside that window is returned.

        Parameters
        ----------
        sample_index : int
            Absolute sample index of the current time step (used to decide when the
            search window is complete).

        candidate_index : int | None
            Candidate index originating from the moving-window integrated signal
            (after peak detection, adaptive thresholding, refractory period and
            optional search-back). This index refers to the MWI timeline.

        fs : float
            Sampling frequency of the ECG signal in Hz.

        search_window : int | None, optional
            Number of samples to search on either side of the candidate.
            Defaults to 15 % of one second (0.15 * fs).

        mwi_delay : int | None, optional
            Approximate delay (in samples) between the filtered signal and the
            moving-window integrated signal. Defaults to ``search_window - 1``.
            This constant can be fine-tuned offline for better sample-level alignment.

        Returns
        -------
        localized_r_peak : float | None
            Amplitude of the localized R-peak. Returns ``None`` while the search
            window is still incomplete.

        localized_r_peak_index : int | None
            Absolute sample index (in the original recording) of the localized R-peak.
            Returns ``None`` while the search window is still incomplete.

        Notes
        -----
        - The filtered signal history must be maintained outside this function:
            filtered_history        → list of amplitude values
            filtered_index_history  → list of absolute sample indices
        - Candidates are stored in the global queue ``pending_candidates`` until
        their complete search window is available.
        - Localization uses the absolute maximum (``np.argmax(np.abs(area))``) so
        both positive and negative R-peaks are correctly identified.
        - The function is designed for real-time, sample-by-sample processing.
        """

        if search_window is None:
            search_window = int(0.15 * fs)

        if mwi_delay is None:
            mwi_delay = search_window - 1

        # New candidate arrived → estimate where it should appear in the filtered timeline
        if candidate_index is not None:
            target_index = candidate_index - mwi_delay
            self.pending_candidates.append((candidate_index, target_index))

        if not self.pending_candidates:
            return None, None

        # Look at the oldest pending candidate
        cand_idx, target_idx = self.pending_candidates[0]

        # Wait until the right side of the search window has arrived
        if sample_index < target_idx + search_window:
            return None, None

        # Locate the target inside the already-stored filtered history
        try:
            center_pos = self.filtered_index_history.index(target_idx)
        except ValueError:
            # Target index never appeared (or already fell out of a limited buffer)
            self.pending_candidates.pop(0)
            return None, None

        lower = max(0, center_pos - search_window)
        upper = min(len(self.filtered_history) - 1, center_pos + search_window)

        area = np.asarray(self.filtered_history[lower : upper + 1])
        local_offset = int(np.argmax(np.abs(area)))
        global_pos = lower + local_offset

        localized_r_peak = float(self.filtered_history[global_pos])
        localized_r_peak_index = self.filtered_index_history[global_pos]

        self.localized_r_peak_hist.append(localized_r_peak)
        self.localized_r_peak_index_hist.append(localized_r_peak_index)

        self.pending_candidates.pop(0)

        return localized_r_peak, localized_r_peak_index

    def _search_back(
        self,
        candidate_peak: float | None,
        candidate_index: int | None,
        fs: float,
        search_back_time: float = 1.66,          # classic Pan-Tompkins value (~1660 ms)
        mwi_delay: int | None = None
    ):
        """
        Perform the online search-back stage of the Pan-Tompkins QRS detection algorithm.

        When a candidate peak fails the primary adaptive threshold (THRESHOLD_1)
        but is still considered, this function looks backward in time (default
        1.66 seconds) for a stronger peak that exceeds the secondary adaptive
        threshold (THRESHOLD_2). If such a peak is found, it replaces the original
        candidate.

        The function maintains its own dedicated history lists so that the results
        of search-back can be inspected and plotted independently of the main
        accepted-peak lists.

        Parameters
        ----------
        candidate_peak : float or None
            Amplitude of the current candidate peak (from the integrated signal).
            Pass None when no candidate is available.

        candidate_index : int or None
            Absolute sample index of the current candidate (MWI timeline).
            Pass None when no candidate is available.

        fs : float
            Sampling frequency of the ECG signal in Hz.

        search_back_time : float, optional
            Maximum time to look backward from the candidate, in seconds.
            Default is 1.66 s (classic Pan-Tompkins value).

        mwi_delay : int or None, optional
            Approximate delay (in samples) between the filtered signal and the
            moving-window integrated signal. Defaults to int(0.15 * fs) - 1.

        Returns
        -------
        final_peak : float or None
            Amplitude of the accepted peak after search-back.
            Returns the original candidate if no better peak is found,
            or None if the input candidate was None.

        final_index : int or None
            Absolute sample index of the accepted peak after search-back.

        is_replaced : bool
            True if a stronger peak was found by search-back and replaced
            the original candidate; False otherwise.

        Notes
        -----
        - The function uses the global adaptive threshold THRESHOLD_2.
        - Only peaks that exceed THRESHOLD_2 are considered valid replacements.
        - Successful replacements are stored in the dedicated lists:
            self.search_back_peaks, self.search_back_indices, self.search_back_flags
        - After a successful search-back the classic Pan-Tompkins algorithm
        updates SPK with the weights 0.25 / 0.75 instead of 0.125 / 0.875.
        That update can be performed by the caller if desired.
        """

        # Handle missing candidate
        if candidate_peak is None or candidate_index is None:
            return None, None, False

        # Protect against THRESHOLD_2 still being None (learning phase)
        if self.THRESHOLD_2 is None:
            return candidate_peak, candidate_index, False

        # Default delay compensation
        if mwi_delay is None:
            mwi_delay = int(0.15 * fs) - 1

        # Number of samples to look back
        N = int(search_back_time * fs)

        # Search region in the filtered-signal timeline
        search_end   = max(0, candidate_index - mwi_delay)
        search_start = max(0, search_end - N)

        # Search for the strongest peak that exceeds THRESHOLD_2
        best_peak  = None
        best_index = None

        for i, idx in enumerate(self.filtered_index_history):
            if idx < search_start:
                continue
            if idx > search_end:
                break

            # Robust amplitude extraction
            amp = self.filtered_history[i] if self.filtered_history[i] is not None else 0.0

            if amp > self.THRESHOLD_2:
                if best_peak is None or abs(amp) > abs(best_peak):
                    best_peak  = amp
                    best_index = idx

        # Decide whether a replacement occurred
        is_replaced = False

        if best_peak is not None:
            self.search_back_peaks.append(best_peak)
            self.search_back_indices.append(best_index)
            self.search_back_flags.append(True)

            final_peak  = best_peak
            final_index = best_index
            is_replaced = True
        else:
            self.search_back_flags.append(False)

            final_peak  = candidate_peak
            final_index = candidate_index
            is_replaced = False

        return final_peak, final_index, is_replaced

    def detect(self, sample: float, sample_index: int, fs: float):
        """
        Process one incoming ECG sample through the complete online
        Pan-Tompkins QRS detection pipeline.

        This is the main entry point of the algorithm. It executes all
        stages in the correct order and returns the localized R-peak
        when one is found.

        Parameters
        ----------
        sample : float
            Current raw ECG sample.
        sample_index : int
            Absolute sample index of the current sample.
        fs : float
            Sampling frequency of the ECG signal in Hz.

        Returns
        -------
        localized_r_peak : float or None
            Amplitude of the localized R-peak. Returns None when no
            R-peak is ready at this time step.
        localized_r_peak_index : int or None
            Absolute sample index of the localized R-peak.
            Returns None when no R-peak is ready.
        """

        # -------------------------------------------------
        # 1. Band-pass filter
        # -------------------------------------------------
        filtered_sample = self._band_pass_filter(sample)

        # Store filtered sample immediately (required by search-back & localization)
        self.filtered_history.append(filtered_sample)
        self.filtered_index_history.append(sample_index)

        # -------------------------------------------------
        # 2. Derivative → Square → Moving-window integration
        # -------------------------------------------------
        derivative_sample = self._derivative(filtered_sample, fs)
        squared_sample = self._square(derivative_sample)
        integrated_sample = self._moving_window_integration(squared_sample, fs)

        # -------------------------------------------------
        # 3. Peak detection on the integrated signal
        # -------------------------------------------------
        loc_peak, loc_pk_index = self._detect_peaks(integrated_sample)

        # -------------------------------------------------
        # 4. Adaptive thresholding
        # -------------------------------------------------
        (
            classified_peak,
            classified_peak_index,
            classified_noise,
            classified_noise_index
        ) = self._adoptive_threshold_online(
            integrated_sample,
            sample_index,
            loc_peak,
            loc_pk_index,
            fs
        )

        # -------------------------------------------------
        # 5. Refractory period
        # -------------------------------------------------
        y_acc_peak, y_acc_peak_index = self._refractory_period(
            classified_peak,
            classified_peak_index,
            fs
        )

        # -------------------------------------------------
        # 6. Search-back
        # -------------------------------------------------
        final_peak, final_index, was_replaced = self._search_back(
            y_acc_peak,
            y_acc_peak_index,
            fs
        )

        # -------------------------------------------------
        # 7. Localize the R-peak in the filtered signal
        # -------------------------------------------------
        localized_r_peak, localized_r_peak_index = self._localize_r_peak(
            sample_index,
            final_index,
            fs
        )

        return localized_r_peak, localized_r_peak_index

