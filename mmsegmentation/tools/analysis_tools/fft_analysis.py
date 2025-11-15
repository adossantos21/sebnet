import argparse
from mmengine.config import Config
from mmseg.apis import init_model
from mmengine.runner import Runner
from utils.frequency_analyzer import FrequencyAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description='Compute frequency percentages for features')
    parser.add_argument('--baseline_config', 
                        type=str, 
                        default='configs/pidnet/pidnet-l_1xb6-241k_1024x1024-cityscapes.py', 
                        help='Path to baseline model config file')
    parser.add_argument('--baseline_checkpoint', 
                        type=str, 
                        default='work_dirs/pidnet-l_1xb6-241k_1024x1024-cityscapes/20251105_161036/checkpoints/best_mIoU.pth' ,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--sbd_config', 
                        type=str, 
                        default='configs/sebnet/sebnet_ablation40_1xb6-160k_cityscapes.py',
                        help='Path to SBD model config file')
    parser.add_argument('--sbd_checkpoint', 
                        type=str, 
                        default='work_dirs/sebnet_ablation40_1xb6-160k_cityscapes/remapped_checkpoint.pth',
                        help='Path to SBD model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0 or cpu)')
    parser.add_argument('--output_metrics', type=str, default='fft_metrics.txt', help='Output file for metrics')
    parser.add_argument('--output_figure_comparison', type=str, default='fft_figure_comparison.png', 
                        help='Output file for comparison figure')
    parser.add_argument('--output_figure_ratio', type=str, default='fft_figure_ratio.png', 
                        help='Output file for ratio figure')
    return parser.parse_args()

def main(args):
    # Load configs
    baseline_cfg = Config.fromfile(args.baseline_config)
    sbd_cfg = Config.fromfile(args.sbd_config)

    # Initialize models
    baseline_model = init_model(baseline_cfg, args.baseline_checkpoint, device=args.device)
    sbd_model = init_model(sbd_cfg, args.sbd_checkpoint, device=args.device)
    
    #print(baseline_model.state_dict().keys(), '\n\n\n')
    #print(sbd_model.state_dict().keys())
    #import sys
    #sys.exit()

    # Build test dataset and dataloader (assuming same test config as baseline)
    test_loader = Runner.build_dataloader(baseline_cfg.val_dataloader)

    # Instantiate analyzer
    analyzer = FrequencyAnalyzer(
        baseline_layer_name='decode_head.i_head.conv.conv', 
        model_layer_name='decode_head.pre_head.conv.conv'
    )

    # Compute metrics and figures
    metrics, fig_comparison, fig_ratio = analyzer.forward(baseline_model, sbd_model, test_loader)

    # Extract statistical significance results
    sig_results = metrics.pop('statistical_significance')

    # Save metrics to txt
    with open(args.output_metrics, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FREQUENCY DOMAIN ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic frequency metrics
        f.write("Frequency Ratio Metrics:\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        
        # Statistical significance section
        f.write("\n" + "=" * 60 + "\n")
        f.write("STATISTICAL SIGNIFICANCE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Sample size: {sig_results['n_samples']}\n")
        f.write(f"Mean high-frequency ratio: {sig_results['mean_ratio']:.6f}\n")
        f.write(f"Standard deviation: {sig_results['std_ratio']:.6f}\n")
        f.write(f"95% Confidence Interval: [{sig_results['ci_95_lower']:.6f}, {sig_results['ci_95_upper']:.6f}]\n")
        f.write(f"\n")
        f.write(f"t-statistic: {sig_results['t_statistic']:.6f}\n")
        f.write(f"p-value: {sig_results['p_value']:.8f}\n")
        f.write(f"Cohen's d (effect size): {sig_results['cohens_d']:.6f}\n")
        f.write(f"\n")
        
        # Interpretation
        f.write("Interpretation:\n")
        f.write("-" * 40 + "\n")
        
        if sig_results['is_significant']:
            f.write("✓ Result is STATISTICALLY SIGNIFICANT (p < 0.05)\n")
            f.write("✓ 95% CI lower bound > 1.0\n")
            improvement_pct = (sig_results['mean_ratio'] - 1.0) * 100
            f.write(f"✓ SBD model retains {improvement_pct:.2f}% more high-frequency information\n")
        else:
            f.write("✗ Result is NOT statistically significant\n")
            if sig_results['p_value'] >= 0.05:
                f.write("  - p-value >= 0.05 (could be due to chance)\n")
            if sig_results['ci_95_lower'] <= 1.0:
                f.write("  - 95% CI includes 1.0 (insufficient evidence of improvement)\n")
        
        f.write(f"\n")
        
        # Effect size interpretation
        f.write("Effect Size Interpretation:\n")
        f.write("-" * 40 + "\n")
        d = abs(sig_results['cohens_d'])
        if d < 0.2:
            effect = "negligible"
        elif d < 0.5:
            effect = "small"
        elif d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        f.write(f"Cohen's d = {sig_results['cohens_d']:.4f} indicates a '{effect}' effect size\n")
        
        f.write("\n" + "=" * 60 + "\n")

    # Save figures
    fig_comparison.savefig(args.output_figure_comparison, dpi=300, bbox_inches='tight')
    fig_ratio.savefig(args.output_figure_ratio, dpi=300, bbox_inches='tight')
    
    print(f"\nAnalysis complete!")
    print(f"Metrics saved to: {args.output_metrics}")
    print(f"Comparison figure saved to: {args.output_figure_comparison}")
    print(f"Ratio figure saved to: {args.output_figure_ratio}")
    
    # Print key results to console
    print(f"\n{'='*60}")
    print(f"KEY RESULTS:")
    print(f"{'='*60}")
    print(f"Mean high-frequency ratio: {sig_results['mean_ratio']:.4f}")
    print(f"95% CI: [{sig_results['ci_95_lower']:.4f}, {sig_results['ci_95_upper']:.4f}]")
    print(f"p-value: {sig_results['p_value']:.6f}")
    print(f"Statistically significant: {'YES' if sig_results['is_significant'] else 'NO'}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)
