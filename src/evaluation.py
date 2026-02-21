"""
Evaluation Module
- Load videos with accident/non-accident labels
- Compute per-video risk statistics
- Generate ROC curves, AUC, precision-recall
- Confusion matrices and performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import json

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
    from sklearn.metrics import classification_report, roc_auc_score
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("scikit-learn and matplotlib required for evaluation. Install via: pip install scikit-learn matplotlib")

logger = logging.getLogger(__name__)


@dataclass
class VideoRiskStats:
    """Per-video risk statistics."""
    video_path: str
    label: int  # 1 = accident, 0 = non-accident
    num_frames: int
    num_agents: int
    
    # Risk statistics (per-agent, then aggregated)
    mean_risk: float  # Mean of all per-frame risks
    max_risk: float   # Maximum risk across all frames
    p95_risk: float   # 95th percentile risk
    std_risk: float   # Std dev of risks
    
    # Peak metrics
    peak_agent_id: int  # Agent with highest risk
    peak_risk: float
    peak_frame: int
    
    # Prediction
    predicted_label: int = 0  # Binary prediction at threshold 0.5
    confidence: float = 0.0   # How confident is the model


class VideoEvaluator:
    """Evaluates video-level accident prediction."""
    
    def __init__(self, risk_threshold: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            risk_threshold: Threshold for binary classification (default 0.5)
        """
        self.risk_threshold = risk_threshold
        self.video_stats: List[VideoRiskStats] = []
    
    def load_risk_results(self, results_csv: str) -> Dict[str, List[float]]:
        """
        Load risk scores from pipeline CSV output.
        
        Args:
            results_csv: Path to risk_scores.csv
            
        Returns:
            dict: video_name -> risk_scores[]
        """
        df = pd.read_csv(results_csv)
        
        # Group by frame and aggregate across agents
        video_risks = {}
        
        for frame_idx, frame_group in df.groupby('frame_idx'):
            # Mean risk across all agents in this frame
            frame_risk = frame_group['smoothed_risk'].mean()
            # Use first filename as key (assumes single video)
            key = 'video'
            if key not in video_risks:
                video_risks[key] = []
            video_risks[key].append(frame_risk)
        
        return video_risks
    
    def compute_video_stats(self, video_path: str, label: int,
                           risk_scores: np.ndarray,
                           frame_info: Optional[Dict] = None) -> VideoRiskStats:
        """
        Compute risk statistics for a single video.
        
        Args:
            video_path: Path to video file
            label: Ground truth (1=accident, 0=non-accident)
            risk_scores: Array of risk scores [0,1] per frame
            frame_info: Optional dict with num_agents, etc.
            
        Returns:
            VideoRiskStats
        """
        risk_scores = np.array(risk_scores)
        
        # Compute statistics
        mean_risk = float(np.mean(risk_scores))
        max_risk = float(np.max(risk_scores))
        p95_risk = float(np.percentile(risk_scores, 95))
        std_risk = float(np.std(risk_scores))
        
        # Find peak
        peak_frame = int(np.argmax(risk_scores))
        peak_risk = float(risk_scores[peak_frame])
        
        # Metadata
        num_frames = len(risk_scores)
        num_agents = frame_info.get('num_agents', -1) if frame_info else -1
        
        # Binary prediction
        predicted_label = 1 if mean_risk >= self.risk_threshold else 0
        confidence = mean_risk if predicted_label == 1 else (1.0 - mean_risk)
        
        stats = VideoRiskStats(
            video_path=str(video_path),
            label=label,
            num_frames=num_frames,
            num_agents=num_agents,
            mean_risk=mean_risk,
            max_risk=max_risk,
            p95_risk=p95_risk,
            std_risk=std_risk,
            peak_agent_id=-1,
            peak_risk=peak_risk,
            peak_frame=peak_frame,
            predicted_label=predicted_label,
            confidence=confidence
        )
        
        self.video_stats.append(stats)
        
        return stats
    
    def get_performance_metrics(self, use_stat: str = 'mean_risk') -> Dict:
        """
        Compute classification metrics using a specific risk statistic.
        
        Args:
            use_stat: Which statistic to use for classification
                     ('mean_risk', 'max_risk', 'p95_risk')
            
        Returns:
            dict: Metrics including AUC, precision, recall, etc.
        """
        if not self.video_stats:
            logger.warning("No video statistics computed yet")
            return {}
        
        # Extract true labels and predicted scores
        y_true = np.array([s.label for s in self.video_stats])
        
        if use_stat == 'mean_risk':
            y_score = np.array([s.mean_risk for s in self.video_stats])
        elif use_stat == 'max_risk':
            y_score = np.array([s.max_risk for s in self.video_stats])
        elif use_stat == 'p95_risk':
            y_score = np.array([s.p95_risk for s in self.video_stats])
        else:
            raise ValueError(f"Unknown statistic: {use_stat}")
        
        # Compute metrics
        try:
            auc_score = roc_auc_score(y_true, y_score)
        except:
            auc_score = 0.0
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        
        # Confusion matrix at default threshold
        y_pred = (y_score >= self.risk_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        metrics = {
            'auc': float(auc_score),
            'accuracy': float(accuracy),
            'precision': float(precision_val),
            'recall': float(recall_val),
            'f1': float(f1),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'roc_fpr': fpr.tolist(),
            'roc_tpr': tpr.tolist(),
            'pr_precision': precision.tolist(),
            'pr_recall': recall.tolist(),
            'statistic_used': use_stat,
            'threshold': self.risk_threshold,
        }
        
        return metrics
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.video_stats:
            logger.warning("No statistics to print")
            return
        
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        # Separate by label
        accidents = [s for s in self.video_stats if s.label == 1]
        non_accidents = [s for s in self.video_stats if s.label == 0]
        
        logger.info(f"Total videos: {len(self.video_stats)}")
        logger.info(f"  Accidents: {len(accidents)}")
        logger.info(f"  Non-accidents: {len(non_accidents)}")
        
        # Risk statistics by class
        if accidents:
            accident_risks = [s.mean_risk for s in accidents]
            logger.info(f"\nAccident videos - Mean risk:")
            logger.info(f"  Mean: {np.mean(accident_risks):.3f}")
            logger.info(f"  Std: {np.std(accident_risks):.3f}")
            logger.info(f"  Min: {np.min(accident_risks):.3f}")
            logger.info(f"  Max: {np.max(accident_risks):.3f}")
        
        if non_accidents:
            non_accident_risks = [s.mean_risk for s in non_accidents]
            logger.info(f"\nNon-accident videos - Mean risk:")
            logger.info(f"  Mean: {np.mean(non_accident_risks):.3f}")
            logger.info(f"  Std: {np.std(non_accident_risks):.3f}")
            logger.info(f"  Min: {np.min(non_accident_risks):.3f}")
            logger.info(f"  Max: {np.max(non_accident_risks):.3f}")
        
        # Performance metrics
        metrics = self.get_performance_metrics(use_stat='mean_risk')
        logger.info(f"\nPerformance Metrics (using mean_risk):")
        logger.info(f"  AUC-ROC: {metrics['auc']:.3f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1']:.3f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}")
        logger.info(f"  FN: {metrics['false_negatives']}, TP: {metrics['true_positives']}")
        logger.info("=" * 80)
    
    def save_results(self, output_dir: str):
        """
        Save evaluation results to CSV and JSON.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save video stats to CSV
        stats_data = []
        for s in self.video_stats:
            stats_data.append({
                'video_path': s.video_path,
                'ground_truth': s.label,
                'predicted': s.predicted_label,
                'num_frames': s.num_frames,
                'num_agents': s.num_agents,
                'mean_risk': s.mean_risk,
                'max_risk': s.max_risk,
                'p95_risk': s.p95_risk,
                'std_risk': s.std_risk,
                'peak_risk': s.peak_risk,
                'peak_frame': s.peak_frame,
                'confidence': s.confidence,
            })
        
        df_stats = pd.DataFrame(stats_data)
        stats_path = output_dir / 'evaluation_results.csv'
        df_stats.to_csv(stats_path, index=False)
        logger.info(f"Saved evaluation results to {stats_path}")
        
        # Save metrics to JSON
        metrics = self.get_performance_metrics(use_stat='mean_risk')
        metrics_path = output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    def plot_roc_curve(self, output_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            output_path: Path to save plot (optional)
        """
        metrics = self.get_performance_metrics(use_stat='mean_risk')
        
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['roc_fpr'], metrics['roc_tpr'], 
                label=f"AUC = {metrics['auc']:.3f}", linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Accident vs Non-Accident Detection', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {output_path}")
        
        return plt
    
    def plot_precision_recall(self, output_path: Optional[str] = None):
        """
        Plot precision-recall curve.
        
        Args:
            output_path: Path to save plot (optional)
        """
        metrics = self.get_performance_metrics(use_stat='mean_risk')
        
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['pr_recall'], metrics['pr_precision'], linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Accident Detection', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {output_path}")
        
        return plt
    
    def plot_risk_distribution(self, output_path: Optional[str] = None):
        """
        Plot distribution of risk scores by class.
        
        Args:
            output_path: Path to save plot (optional)
        """
        accidents = [s.mean_risk for s in self.video_stats if s.label == 1]
        non_accidents = [s.mean_risk for s in self.video_stats if s.label == 0]
        
        plt.figure(figsize=(12, 6))
        
        if accidents:
            plt.hist(accidents, bins=10, alpha=0.7, label='Accidents', color='red')
        if non_accidents:
            plt.hist(non_accidents, bins=10, alpha=0.7, label='Non-Accidents', color='green')
        
        plt.xlabel('Mean Risk Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Risk Scores by Class', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved risk distribution to {output_path}")
        
        return plt
    
    def plot_confusion_matrix(self, output_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            output_path: Path to save plot (optional)
        """
        metrics = self.get_performance_metrics(use_stat='mean_risk')
        
        cm = np.array([
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ])
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                        color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
        
        plt.xticks([0, 1], ['Non-Accident', 'Accident'])
        plt.yticks([0, 1], ['Non-Accident', 'Accident'])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Ground Truth', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {output_path}")
        
        return plt
