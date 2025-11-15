import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy import fftpack, stats
from tqdm import tqdm

class FrequencyAnalyzer:
    def __init__(self, baseline_layer_name='backbone.stage3', model_layer_name='backbone.stage3'):
        self.baseline_layer_name = baseline_layer_name
        self.model_layer_name = model_layer_name
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_feature_maps(self, model, image, baseline=False):
        features = {}
        def hook_fn(module, input, output):
            features['output'] = output.detach()
        
        # Register hook
        layer_name = self.baseline_layer_name if baseline else self.model_layer_name
        layer_found = False
        for name, module in model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                layer_found = True
                break
        
        if not layer_found:
            print(f"Layer '{layer_name}' not found. Available layers:")
            for name, _ in model.named_modules():
                if name:
                    print(f"  - {name}")
            raise ValueError(f"Layer '{layer_name}' not found in model")

        with torch.no_grad():
            _ = model(image)
        
        handle.remove()
        return features['output']

    def compute_frequency_spectrum(self, feature_maps):
        features = feature_maps.cpu().numpy() # has shape (N, C, H, W)
        fft_per_channel = []
        for b in range(features.shape[0]):
            for c in range(features.shape[1]):
                fft = fftpack.fft2(features[b, c])
                fft_magnitude = np.abs(fftpack.fftshift(fft))
                fft_per_channel.append(fft_magnitude)
        avg_fft_magnitude = np.mean(fft_per_channel, axis=0)
        
        h, w = avg_fft_magnitude.shape
        cy, cx = h//2, w//2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        
        radial_prof = np.bincount(r.ravel(), avg_fft_magnitude.ravel()) / np.bincount(r.ravel())
        
        return radial_prof

    def compare_frequency_preservation(self, baseline_model, sbd_model, test_loader:torch.utils.data.DataLoader):
        
        baseline_spectra = []
        sbd_spectra = []
        for data in tqdm(test_loader, desc='Processing data'):
            inputs = data['inputs'][0].to(torch.float32) / 255.0

            images = self.transform(inputs).unsqueeze(0).cuda() # inputs has shape (C, H, W)

            baseline_features = self.extract_feature_maps(baseline_model, images, baseline=True)
            sbd_features = self.extract_feature_maps(sbd_model, images, baseline=False)
            
            baseline_spectrum = self.compute_frequency_spectrum(baseline_features)
            sbd_spectrum = self.compute_frequency_spectrum(sbd_features)
            
            baseline_spectra.append(baseline_spectrum)
            sbd_spectra.append(sbd_spectrum)
        
        # Return individual spectra for statistical analysis
        return baseline_spectra, sbd_spectra

    def compute_statistical_significance(self, baseline_spectra, sbd_spectra):
        """
        Compute statistical significance of high-frequency ratio improvement.
        Tests null hypothesis that mean ratio = 1.0 (no improvement).
        """
        per_sample_ratios = []
        
        for baseline_spec, sbd_spec in zip(baseline_spectra, sbd_spectra):
            # Normalize each spectrum by its DC component
            norm_baseline = baseline_spec / (baseline_spec[0] + 1e-10)
            norm_sbd = sbd_spec / (sbd_spec[0] + 1e-10)
            
            # Define high frequency range (1/4 to 1/2 of max frequency)
            max_freq = len(norm_baseline)
            high_freq_start = max_freq // 4
            high_freq_end = max_freq // 2
            
            # Compute mean high-freq ratio for this sample
            ratio = np.mean(norm_sbd[high_freq_start:high_freq_end] / 
                           (norm_baseline[high_freq_start:high_freq_end] + 1e-10))
            per_sample_ratios.append(ratio)
        
        ratios = np.array(per_sample_ratios)
        
        # One-sample t-test: is mean ratio significantly > 1.0?
        t_stat, p_value = stats.ttest_1samp(ratios, 1.0, alternative='greater')
        
        # Confidence interval (95%)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios, ddof=1)
        n = len(ratios)
        se = std_ratio / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, loc=mean_ratio, scale=se)
        
        # Cohen's d effect size
        cohens_d = (mean_ratio - 1.0) / std_ratio
        
        # Determine significance
        is_significant = (p_value < 0.05) and (ci_95[0] > 1.0)
        
        significance_results = {
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'n_samples': n,
            'is_significant': is_significant,
            'per_sample_ratios': ratios  # For debugging/further analysis
        }
        
        return significance_results

    def plot_frequency_analysis(self, baseline_spectrum, sbd_spectrum):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        max_freq = min(len(baseline_spectrum), len(sbd_spectrum))
        freqs = np.arange(max_freq)
        
        axes[0].semilogy(freqs[:max_freq//2], baseline_spectrum[:max_freq//2], 
                         label='PIDNet-L (Baseline)', linewidth=2)
        axes[0].semilogy(freqs[:max_freq//2], sbd_spectrum[:max_freq//2], 
                         label='PIDNet-L+SBD', linewidth=2)
        axes[0].set_xlabel('Spatial Frequency')
        axes[0].set_ylabel('Normalized Magnitude (log scale)')
        axes[0].set_title('Frequency Spectrum Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        high_freq_start = max_freq // 4
        ratio = sbd_spectrum[high_freq_start:max_freq//2] / baseline_spectrum[high_freq_start:max_freq//2]
        
        axes[1].plot(freqs[high_freq_start:max_freq//2], ratio, linewidth=2, color='green')
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Spatial Frequency (High Frequency Range)')
        axes[1].set_ylabel('SBD/Baseline Ratio')
        axes[1].set_title('High Frequency Retention Improvement')
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(freqs[high_freq_start:max_freq//2], 1, ratio, 
                            where=(ratio > 1), alpha=0.3, color='green', 
                            label='Improvement Region')
        axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_frequency_analysis_ratio(self, baseline_spectrum, sbd_spectrum):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        
        max_freq = min(len(baseline_spectrum), len(sbd_spectrum))
        freqs = np.arange(max_freq)
        
        high_freq_start = max_freq // 4
        ratio = sbd_spectrum[high_freq_start:max_freq//2] / baseline_spectrum[high_freq_start:max_freq//2]
        
        ax.plot(freqs[high_freq_start:max_freq//2], ratio, linewidth=2, color='green')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Spatial Frequency (High Frequency Range)')
        ax.set_ylabel('SBD/Baseline Ratio')
        ax.set_title('High Frequency Retention Improvement')
        ax.grid(True, alpha=0.3)
        ax.fill_between(freqs[high_freq_start:max_freq//2], 1, ratio, 
                        where=(ratio > 1), alpha=0.3, color='green', 
                        label='Improvement Region')
        ax.legend()
        
        plt.tight_layout()
        return fig

    def compute_frequency_metrics(self, baseline_spectrum, sbd_spectrum):
        
        total_len = len(baseline_spectrum)
        low_freq = total_len // 8
        mid_freq = total_len // 4
        high_freq = total_len // 2
        
        metrics = {
            'high_freq_ratio': np.mean(sbd_spectrum[mid_freq:high_freq] / 
                                       (baseline_spectrum[mid_freq:high_freq] + 1e-10)),
            'mid_freq_ratio': np.mean(sbd_spectrum[low_freq:mid_freq] / 
                                      (baseline_spectrum[low_freq:mid_freq]+ 1e-10)),
            'low_freq_ratio': np.mean(sbd_spectrum[:low_freq] /
                              (baseline_spectrum[:low_freq]+ 1e-10)),
        }
        
        return metrics

    def forward(self, baseline_model, sbd_model, test_loader):
        
        # Get individual spectra for statistical testing
        baseline_spectra, sbd_spectra = self.compare_frequency_preservation(
            baseline_model, sbd_model, test_loader)
        
        # Compute statistical significance
        significance_results = self.compute_statistical_significance(
            baseline_spectra, sbd_spectra)
        
        # Compute averaged and normalized spectra for plotting
        avg_baseline = np.mean(baseline_spectra, axis=0)
        avg_sbd = np.mean(sbd_spectra, axis=0)
        norm_avg_baseline = avg_baseline / avg_baseline[0]
        norm_avg_sbd = avg_sbd / avg_sbd[0]
        
        # Compute frequency metrics
        metrics = self.compute_frequency_metrics(norm_avg_baseline, norm_avg_sbd)
        
        # Add significance results to metrics
        metrics.update({
            'statistical_significance': significance_results
        })
        
        # Generate plots
        fig_comparison = self.plot_frequency_analysis(norm_avg_baseline, norm_avg_sbd)
        fig_ratio = self.plot_frequency_analysis_ratio(norm_avg_baseline, norm_avg_sbd)
        
        return metrics, fig_comparison, fig_ratio
