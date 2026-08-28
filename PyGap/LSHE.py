import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, lstsq
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution

class LSHE:
    """
    Least Squares Harmonic Estimation (LS-HE) Estimator (Amiri-Simkooei style).
    
    Fits a linear baseline model (intercept + linear trend) and iteratively 
    extracts statistically significant frequencies conditional on previous selections.
    """
    
    def __init__(self, max_freqs=10, ofac=5, alpha=0.05):
        """
        Initializes the LS-HE model.
        
        Args:
            max_freqs (int): Maximum harmonic frequencies to search for.
            ofac (int): Oversampling grid multiplier factor.
            alpha (float): F-test significance type-I threshold error limit.
        """
        self.max_freqs = max_freqs
        self.ofac = ofac
        self.alpha = alpha
        
        # Parameters estimated after running .fit()
        self.frequencies_ = []
        self.weights_ = None
        self.sigma2_ = None
        self.dof_ = None
        self.q_cofactor_ = None
        self.log_sq_weights_ = []
        
        # Internal reference data caches
        self._x_offset = None
        self._a_design_fit = None

    def _build_ak(self, t, omega):
        """Design matrix for a single frequency [cos(w*t), sin(w*t)]."""
        return np.column_stack([np.cos(omega * t), np.sin(omega * t)])

    def _projection_matrix(self, a, p):
        """Computes structural projection components P_A and its orthogonal complement P_A^perp."""
        n_inv = inv(a.T @ p @ a)
        pa = a @ n_inv @ a.T @ p
        p_perp = np.eye(a.shape[0]) - pa
        return pa, p_perp

    def fit(self, x, y, p=None):
        """
        Trains the LS-HE parameters by locating conditional spectral frequencies.
        
        Args:
            x (array-like): Input time coordinate track.
            y (array-like): Observed tracking parameters measurements.
            p (array-like, optional): Matrix weights configuration (Defaults to Identity).
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples = len(x_arr)
        
        # Normalization tracking
        self._x_offset = np.min(x_arr)
        x_norm = x_arr - self._x_offset
        
        if p is None:
            p = np.eye(n_samples)
            
        # Initialize base trends array structural model: intercept + slope linear trend
        a_base = np.column_stack([np.ones(n_samples), x_norm])
        
        # Define the search frequency grid
        t_span = np.max(x_norm)
        dt = np.diff(np.sort(x_norm))
        f_min = 1.0 / t_span
        f_max = 1.0 / (2.0 * np.median(dt))
        n_freqs = int(self.ofac * n_samples)
        freq_grid = np.linspace(f_min, f_max, n_freqs)
        
        self.frequencies_ = []
        
        # Iterative Frequency Detection Loop
        for i in range(self.max_freqs):
            # 1. Build current active design space framework
            if self.frequencies_:
                a_freq = np.hstack([self._build_ak(x_norm, 2 * np.pi * f) for f in self.frequencies_])
                a_current = np.hstack([a_base, a_freq])
            else:
                a_current = a_base
                
            # 2. Project observations into the orthogonal complement residual space
            _, p_perp = self._projection_matrix(a_current, p)
            
            best_pow = -1.0
            f_best = None
            
            # 3. Frequency Spectrum Search: Identify the highest projecting harmonic element
            for f_test in freq_grid:
                if f_test in self.frequencies_:
                    continue
                ak = self._build_ak(x_norm, 2 * np.pi * f_test)
                m = ak.T @ p_perp @ ak
                try:
                    pow_val = y_arr.T @ p_perp @ ak @ inv(m) @ ak.T @ p_perp @ y_arr
                    if pow_val > best_pow:
                        best_pow = pow_val
                        f_best = f_test
                except np.linalg.LinAlgError:
                    continue
            
            # 4. Statistical Validation: Validate the candidate via F-Test
            coef, *_ = lstsq(a_current, y_arr, rcond=None)
            current_residuals = y_arr - a_current @ coef
            current_dof = n_samples - a_current.shape[1]
            sigma2_current = (current_residuals.T @ p @ current_residuals) / current_dof
            
            f_obs = (best_pow / 2.0) / sigma2_current
            f_crit = f_distribution.ppf(1.0 - self.alpha, 2, current_dof)
            
            # Conditional extraction stops if the component is statistically insignificant
            if f_obs > f_crit and f_best is not None:
                self.frequencies_.append(f_best)
            else:
                break
                
        # Final parameters extraction phase using the completed structural framework
        if self.frequencies_:
            a_final = np.hstack([a_base] + [self._build_ak(x_norm, 2 * np.pi * f) for f in self.frequencies_])
        else:
            a_final = a_base
            
        self._a_design_fit = a_final
        self.q_cofactor_ = inv(a_final.T @ p @ a_final)
        self.weights_ = self.q_cofactor_ @ a_final.T @ p @ y_arr
        
        final_residuals = y_arr - a_final @ self.weights_
        self.dof_ = n_samples - a_final.shape[1]
        self.sigma2_ = (final_residuals.T @ p @ final_residuals) / self.dof_
        
        # Compute log(square of weights) for the detected harmonic frequencies
        self.log_sq_weights_ = []
        for idx in range(len(self.frequencies_)):
            # Offset by 2 to account for the intercept and linear slope coefficients
            w_cos = self.weights_[2 + 2 * idx]
            w_sin = self.weights_[2 + 2 * idx + 1]
            amplitude_sq = w_cos**2 + w_sin**2
            self.log_sq_weights_.append(np.log10(amplitude_sq) if amplitude_sq > 0 else -12.0)
            
        return self

    def predict(self, x):
        """Predicts signal target values for input timelines."""
        if self.weights_ is None:
            raise ValueError("Run .fit() before predicting.")
        x_norm = np.asarray(x, dtype=float) - self._x_offset
        a_pred = np.column_stack([np.ones(len(x_norm)), x_norm])
        if self.frequencies_:
            a_freq = np.hstack([self._build_ak(x_norm, 2 * np.pi * f) for f in self.frequencies_])
            a_pred = np.hstack([a_pred, a_freq])
        return a_pred @ self.weights_

    def confidence_interval(self, x, confidence=0.95):
        """Computes prediction lower and upper uncertainty intervals boundary frames."""
        if self.weights_ is None:
            raise ValueError("Run .fit() before calculating confidence intervals.")
        x_norm = np.asarray(x, dtype=float) - self._x_offset
        a_pred = np.column_stack([np.ones(len(x_norm)), x_norm])
        if self.frequencies_:
            a_freq = np.hstack([self._build_ak(x_norm, 2 * np.pi * f) for f in self.frequencies_])
            a_pred = np.hstack([a_pred, a_freq])
            
        var_y = np.sum((a_pred @ self.q_cofactor_) * a_pred, axis=1) * self.sigma2_
        std_y = np.sqrt(var_y)
        
        t_crit = t_distribution.ppf(1.0 - (1.0 - confidence) / 2.0, self.dof_)
        y_pred = a_pred @ self.weights_
        
        return y_pred - t_crit * std_y, y_pred + t_crit * std_y
    
def simulateData(n_points=200):
     ''' Synthesize verification timeline benchmarks '''
            # 1. 
     np.random.seed(0)     
     x_data = np.sort(np.random.rand(n_points) * 10)
     true_frequencies = [0.5, 1.2, 2.0]
    
     y_clean = np.zeros_like(x_data)
     for freq in true_frequencies:
        y_clean += np.random.uniform(1, 2) * np.cos(2 * np.pi * freq * x_data + np.random.uniform(0, 2 * np.pi))
     y_data = y_clean + 0.5 * np.random.randn(n_points)

     return x_data, y_data, true_frequencies

# Verification execution module
if __name__ == '__main__':
    # 1. Synthesize verification timeline benchmarks
    np.random.seed(0)
    n_points = 200
    x_data = np.sort(np.random.rand(n_points) * 10)
    true_frequencies = [0.5, 1.2, 2.0]
    
    y_clean = np.zeros_like(x_data)
    for freq in true_frequencies:
        y_clean += np.random.uniform(1, 2) * np.cos(2 * np.pi * freq * x_data + np.random.uniform(0, 2 * np.pi))
    y_data = y_clean + 0.5 * np.random.randn(n_points)
    
    # 2. Run model lifecycle
    xx_range = np.linspace(min(x_data), max(x_data), 500)
    estimator = LSHE(max_freqs=10, ofac=5, alpha=0.05)
    estimator.fit(x_data, y_data)
    
    y_pred = estimator.predict(xx_range)
    lower_bound, upper_bound = estimator.confidence_interval(xx_range, confidence=0.95)
    
    # 3. Chart (1): Time domain visualization
    plt.figure(figsize=(12, 5))
    plt.scatter(x_data, y_data, color='black', marker='o', s=15, label='Known Data (x, y)')
    plt.plot(xx_range, y_pred, color='red', linestyle='-', linewidth=0.85, label='LS-HE Prediction')
    plt.fill_between(xx_range, lower_bound, upper_bound, color='black', alpha=0.75, label='95% Confidence Band')
    plt.title('LS-HE Time Domain Interpolation & Prediction Intervals')
    plt.xlabel('Time (x)')
    plt.ylabel('Signal (y)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    # 4. Chart (2): Spectral Domain Matrix Weight Profiling
    plt.figure(figsize=(12, 4))
    est_freqs = estimator.frequencies_
    log_weights = estimator.log_sq_weights_
    
    if est_freqs:
        # Sort values together sequentially along frequency paths for clean dashed connections
        sorted_indices = np.argsort(est_freqs)
        sorted_freqs = np.array(est_freqs)[sorted_indices]
        sorted_weights = np.array(log_weights)[sorted_indices]
        
        plt.plot(sorted_freqs, sorted_weights, color='blue', linestyle='--', marker='s', label='Estimated Frequencies')
    
    #output the linear components
    # Extract parameters directly from the fitted model
    intercept = estimator.weights_[0]
    slope = estimator.weights_[1]

    print(f"Baseline Intercept: {intercept:.4f}")
    print(f"Linear Slope of Change: {slope:.4f} units per time-step")

    # Draw reference markings highlighting known baseline targets
    for tf in true_frequencies:
        plt.axvline(x=tf, color='green', linestyle=':', alpha=0.8, label=f'True Freq ({tf} Hz)' if tf == true_frequencies[0] else "")
        
    plt.title('LS-HE Spectrum: Detected log10(Weight^2) vs Frequencies')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('log10(Power of Amplitude Weight Coefficients)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()