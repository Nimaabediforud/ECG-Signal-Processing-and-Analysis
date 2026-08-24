from preprocessing import PanTompkinsPreprocessor
import numpy as np


class PanTompkinsOffline(PanTompkinsPreprocessor):
    def __init__(self):
        pass

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
    

    