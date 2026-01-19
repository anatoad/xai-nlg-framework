import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def generate_visualization_from_file(results_csv_path):
    try:
        logger.info("Generating visualizations from file...")

        # Load results from CSV
        results_df = pd.read_csv(results_csv_path)

        # Clarity distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist(results_df['clarity_score'], bins=15, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Clarity Score Distribution')
        axes[0, 0].set_xlabel('Clarity Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(45, color='red', linestyle='--', label='Min Threshold')
        axes[0, 0].legend()

        # Coverage
        axes[0, 1].hist(results_df['coverage_topk'], bins=15, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Top-K Coverage Distribution')
        axes[0, 1].set_xlabel('Coverage (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(80, color='red', linestyle='--', label='Target: 80%')
        axes[0, 1].legend()

        # Fidelity
        axes[1, 0].hist(results_df['fidelity_score'], bins=15, color='lightyellow', edgecolor='black')
        axes[1, 0].set_title('Fidelity Score Distribution')
        axes[1, 0].set_xlabel('Fidelity Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(0.8, color='red', linestyle='--', label='Target: 0.8')
        axes[1, 0].legend()

        # Runtime
        axes[1, 1].boxplot(results_df['runtime_ms'], labels=['Runtime'])
        axes[1, 1].set_title('Runtime Distribution')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].axhline(5000, color='red', linestyle='--', label='Budget: 5s')
        axes[1, 1].legend()

        plt.tight_layout()
        fig.savefig('./evaluation_results/metrics_distribution.png', dpi=150)
        logger.info("Visualization saved to ./evaluation_results/metrics_distribution.png")

        # Comparison by method
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df = results_df.groupby('method')[['clarity_score', 'coverage_topk', 'fidelity_score']].mean()
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title('Average Metrics by Method')
        ax.set_ylabel('Score')
        ax.legend(loc='best')
        plt.tight_layout()
        fig.savefig('./evaluation_results/method_comparison.png', dpi=150)
        logger.info("Method comparison saved to ./evaluation_results/method_comparison.png")

    except ImportError:
        logger.warning("Matplotlib or Seaborn not available, skipping visualizations")
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")