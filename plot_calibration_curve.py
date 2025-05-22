import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_prob, output_path, bin_count=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bin_count)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted confidence')
    plt.ylabel('Fraction correct')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot calibration curve from scores.csv using self_conf and score columns. Correctness is 1 if score >= 8, else 0.")
    parser.add_argument('--csv', type=str, default='scores.csv', help='Path to CSV file')
    parser.add_argument('--output', type=str, default='calibration_curve.png', help='Output PNG file')
    parser.add_argument('--bin_count', type=int, default=10, help='Number of bins for calibration curve')
    parser.add_argument('--threshold', type=float, default=8.0, help='Score threshold for correctness (default: 8.0)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Use self_conf as confidence (normalize to 0-1)
    y_prob = df['self_conf'].values / 100.0
    # Use score >= threshold as correct (1), else 0
    y_true = (df['score'].values >= args.threshold).astype(int)
    plot_calibration_curve(y_true, y_prob, args.output, bin_count=args.bin_count)
    print(f"Calibration curve saved to {args.output}")


if __name__ == "__main__":
    main() 