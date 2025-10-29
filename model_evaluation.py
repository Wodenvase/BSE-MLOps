#!/usr/bin/env python3
"""
Model Evaluation Script for SENSEX ConvLSTM - Phase 2
Comprehensive model evaluation with detailed analysis and reporting
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Any
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras

# MLflow for logging
import mlflow
import mlflow.tensorflow

# DVC for data loading
import dvc.api

# Local imports
sys.path.insert(0, '/app/src')
from models.convlstm_model import ConvLSTMModel, ConvLSTMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/model_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SensexModelEvaluator:
    """
    Comprehensive model evaluation for SENSEX ConvLSTM
    """
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.X_test = None
        self.y_test = None
        self.dates_test = None
        self.predictions = None
        self.probabilities = None
        
        # Load model and data
        self.load_model()
        self.load_test_data()
    
    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"Model summary: {self.model.count_params():,} parameters")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def load_test_data(self):
        """Load test data for evaluation"""
        logger.info("Loading test data...")
        
        try:
            # Load from DVC
            with dvc.api.open('data/processed/feature_maps.npy', mode='rb') as f:
                feature_maps = np.load(f)
            
            with dvc.api.open('data/processed/targets.npy', mode='rb') as f:
                targets = np.load(f)
            
            with dvc.api.open('data/processed/dates.csv') as f:
                dates_df = pd.read_csv(f)
                dates = dates_df['date'].tolist()
            
        except Exception as e:
            logger.warning(f"Failed to load from DVC: {e}. Trying local storage...")
            
            # Fallback to local
            data_path = Path('/app/data/processed')
            feature_maps = np.load(data_path / 'feature_maps.npy')
            targets = np.load(data_path / 'targets.npy')
            
            dates_df = pd.read_csv(data_path / 'dates.csv')
            dates = dates_df['date'].tolist()
        
        # Prepare sequences
        temp_config = ConvLSTMConfig()
        temp_model = ConvLSTMModel(temp_config)
        X, y = temp_model.prepare_data(feature_maps, targets)
        
        # Get test split (same as training)
        train_size = int(len(X) * self.config['data_split']['train_ratio'])
        val_size = int(len(X) * self.config['data_split']['val_ratio'])
        
        self.X_test = X[train_size + val_size:]
        self.y_test = y[train_size + val_size:]
        
        # Get corresponding dates for test set
        sequence_length = temp_config.sequence_length
        test_start_idx = train_size + val_size + sequence_length
        self.dates_test = dates[test_start_idx:test_start_idx + len(self.y_test)]
        
        logger.info(f"Test data loaded: {self.X_test.shape[0]} samples")
        logger.info(f"Date range: {self.dates_test[0]} to {self.dates_test[-1]}")
    
    def generate_predictions(self):
        """Generate predictions on test data"""
        logger.info("Generating predictions...")
        
        # Get predictions
        self.probabilities = self.model.predict(self.X_test).flatten()
        self.predictions = (self.probabilities > 0.5).astype(int)
        
        logger.info(f"Predictions generated for {len(self.predictions)} samples")
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        logger.info("Calculating evaluation metrics...")
        
        if self.predictions is None:
            self.generate_predictions()
        
        # Basic classification metrics
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = recall_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)
        
        # ROC metrics
        roc_auc = roc_auc_score(self.y_test, self.probabilities)
        
        # Precision-Recall metrics
        avg_precision = average_precision_score(self.y_test, self.probabilities)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Trading-specific metrics
        up_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0  # Accuracy on up days
        down_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0  # Accuracy on down days
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'specificity': specificity,
            'npv': npv,  # Negative Predictive Value
            'up_day_accuracy': up_accuracy,
            'down_day_accuracy': down_accuracy,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(self.y_test),
            'positive_samples': int(np.sum(self.y_test)),
            'negative_samples': int(len(self.y_test) - np.sum(self.y_test))
        }
        
        logger.info("📊 Evaluation Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Up Day Accuracy: {up_accuracy:.4f}")
        logger.info(f"  Down Day Accuracy: {down_accuracy:.4f}")
        
        return metrics
    
    def create_visualizations(self, output_dir: str) -> Dict[str, str]:
        """Create comprehensive visualization plots"""
        logger.info("Creating evaluation visualizations...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.predictions is None:
            self.generate_predictions()
        
        visualization_paths = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title('Confusion Matrix - SENSEX Direction Prediction')
        plt.ylabel('Actual Direction')
        plt.xlabel('Predicted Direction')
        
        cm_path = f"{output_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['confusion_matrix'] = cm_path
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
        roc_auc = roc_auc_score(self.y_test, self.probabilities)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - SENSEX Direction Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        roc_path = f"{output_dir}/roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['roc_curve'] = roc_path
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, self.probabilities)
        avg_precision = average_precision_score(self.y_test, self.probabilities)
        
        plt.plot(recall_curve, precision_curve, color='blue', lw=2,
                label=f'PR Curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - SENSEX Direction Prediction')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        pr_path = f"{output_dir}/precision_recall_curve.png"
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['precision_recall_curve'] = pr_path
        
        # 4. Prediction Distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.probabilities[self.y_test == 0], bins=20, alpha=0.7, 
                label='Down Days', color='red', density=True)
        plt.hist(self.probabilities[self.y_test == 1], bins=20, alpha=0.7, 
                label='Up Days', color='green', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.axvline(0.5, color='black', linestyle='--', alpha=0.8, label='Threshold')
        
        plt.subplot(1, 2, 2)
        confidence = np.abs(self.probabilities - 0.5) * 2  # Confidence score
        correct = (self.predictions == self.y_test)
        
        plt.scatter(confidence[correct], self.probabilities[correct], 
                   alpha=0.6, c='green', label='Correct', s=10)
        plt.scatter(confidence[~correct], self.probabilities[~correct], 
                   alpha=0.6, c='red', label='Incorrect', s=10)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Predicted Probability')
        plt.title('Prediction Confidence vs Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        dist_path = f"{output_dir}/prediction_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['prediction_distribution'] = dist_path
        
        # 5. Time Series Analysis
        plt.figure(figsize=(15, 8))
        
        # Convert dates to datetime
        dates_dt = pd.to_datetime(self.dates_test)
        
        plt.subplot(2, 1, 1)
        colors = ['red' if pred == 0 else 'green' for pred in self.predictions]
        markers = ['v' if actual != pred else 'o' for actual, pred in zip(self.y_test, self.predictions)]
        
        for i, (date, prob, actual, pred) in enumerate(zip(dates_dt, self.probabilities, self.y_test, self.predictions)):
            color = 'green' if actual == 1 else 'red'
            marker = 'o' if actual == pred else 'x'
            plt.scatter(date, prob, c=color, marker=marker, alpha=0.7, s=30)
        
        plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Predicted Probability')
        plt.title('Prediction Timeline - SENSEX Direction')
        plt.grid(True, alpha=0.3)
        plt.legend(['Threshold', 'Up (Correct)', 'Down (Correct)', 'Up (Wrong)', 'Down (Wrong)'])
        
        plt.subplot(2, 1, 2)
        # Rolling accuracy
        window = 30  # 30-day rolling window
        if len(self.predictions) >= window:
            rolling_accuracy = pd.Series(self.predictions == self.y_test).rolling(window).mean()
            plt.plot(dates_dt, rolling_accuracy, color='blue', linewidth=2)
            plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
            plt.ylabel('Rolling Accuracy (30-day)')
            plt.xlabel('Date')
            plt.title('Model Performance Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        
        timeline_path = f"{output_dir}/timeline_analysis.png"
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['timeline_analysis'] = timeline_path
        
        logger.info(f"Visualizations saved to: {output_dir}")
        
        return visualization_paths
    
    def generate_detailed_report(self, metrics: Dict, output_dir: str) -> str:
        """Generate detailed evaluation report"""
        logger.info("Generating detailed evaluation report...")
        
        report_path = f"{output_dir}/evaluation_report.html"
        
        # Create detailed analysis
        correct_predictions = self.predictions == self.y_test
        high_confidence = np.abs(self.probabilities - 0.5) > 0.3
        
        # Performance by confidence
        high_conf_accuracy = np.mean(correct_predictions[high_confidence]) if np.any(high_confidence) else 0
        low_conf_accuracy = np.mean(correct_predictions[~high_confidence]) if np.any(~high_confidence) else 0
        
        # Performance by market direction
        up_days_mask = self.y_test == 1
        down_days_mask = self.y_test == 0
        
        up_days_pred_accuracy = np.mean(correct_predictions[up_days_mask]) if np.any(up_days_mask) else 0
        down_days_pred_accuracy = np.mean(correct_predictions[down_days_mask]) if np.any(down_days_mask) else 0
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SENSEX ConvLSTM Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .section {{ margin: 30px 0; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .good {{ color: green; font-weight: bold; }}
                .fair {{ color: orange; font-weight: bold; }}
                .poor {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SENSEX ConvLSTM Model Evaluation Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Model:</strong> {self.model_path}</p>
                <p><strong>Test Period:</strong> {self.dates_test[0]} to {self.dates_test[-1]}</p>
                <p><strong>Test Samples:</strong> {len(self.y_test)}</p>
            </div>
            
            <div class="section">
                <h2>📊 Key Performance Metrics</h2>
                <table class="table">
                    <tr><th>Metric</th><th>Value</th><th>Rating</th></tr>
                    <tr><td>Accuracy</td><td>{metrics['accuracy']:.4f}</td><td class="{'good' if metrics['accuracy'] > 0.6 else 'fair' if metrics['accuracy'] > 0.55 else 'poor'}">{self._rate_metric(metrics['accuracy'], 0.6, 0.55)}</td></tr>
                    <tr><td>Precision</td><td>{metrics['precision']:.4f}</td><td class="{'good' if metrics['precision'] > 0.6 else 'fair' if metrics['precision'] > 0.55 else 'poor'}">{self._rate_metric(metrics['precision'], 0.6, 0.55)}</td></tr>
                    <tr><td>Recall</td><td>{metrics['recall']:.4f}</td><td class="{'good' if metrics['recall'] > 0.6 else 'fair' if metrics['recall'] > 0.55 else 'poor'}">{self._rate_metric(metrics['recall'], 0.6, 0.55)}</td></tr>
                    <tr><td>F1-Score</td><td>{metrics['f1_score']:.4f}</td><td class="{'good' if metrics['f1_score'] > 0.6 else 'fair' if metrics['f1_score'] > 0.55 else 'poor'}">{self._rate_metric(metrics['f1_score'], 0.6, 0.55)}</td></tr>
                    <tr><td>ROC-AUC</td><td>{metrics['roc_auc']:.4f}</td><td class="{'good' if metrics['roc_auc'] > 0.65 else 'fair' if metrics['roc_auc'] > 0.55 else 'poor'}">{self._rate_metric(metrics['roc_auc'], 0.65, 0.55)}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Trading Performance Analysis</h2>
                <table class="table">
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                    <tr><td>Up Day Accuracy</td><td>{metrics['up_day_accuracy']:.4f}</td><td>Accuracy in predicting market up days</td></tr>
                    <tr><td>Down Day Accuracy</td><td>{metrics['down_day_accuracy']:.4f}</td><td>Accuracy in predicting market down days</td></tr>
                    <tr><td>High Confidence Accuracy</td><td>{high_conf_accuracy:.4f}</td><td>Accuracy when model is highly confident (>70%)</td></tr>
                    <tr><td>Low Confidence Accuracy</td><td>{low_conf_accuracy:.4f}</td><td>Accuracy when model is less confident (≤70%)</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🔢 Confusion Matrix Analysis</h2>
                <table class="table">
                    <tr><th></th><th>Predicted Down</th><th>Predicted Up</th><th>Total</th></tr>
                    <tr><th>Actual Down</th><td>{metrics['true_negatives']}</td><td>{metrics['false_positives']}</td><td>{metrics['negative_samples']}</td></tr>
                    <tr><th>Actual Up</th><td>{metrics['false_negatives']}</td><td>{metrics['true_positives']}</td><td>{metrics['positive_samples']}</td></tr>
                    <tr><th>Total</th><td>{metrics['true_negatives'] + metrics['false_negatives']}</td><td>{metrics['false_positives'] + metrics['true_positives']}</td><td>{metrics['total_samples']}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🎯 Model Interpretation</h2>
                <h3>Strengths:</h3>
                <ul>
                    {self._generate_strengths(metrics)}
                </ul>
                
                <h3>Areas for Improvement:</h3>
                <ul>
                    {self._generate_improvements(metrics)}
                </ul>
                
                <h3>Trading Implications:</h3>
                <ul>
                    {self._generate_trading_insights(metrics, high_conf_accuracy)}
                </ul>
            </div>
            
            <div class="section">
                <h2>📁 Generated Artifacts</h2>
                <ul>
                    <li>Confusion Matrix: confusion_matrix.png</li>
                    <li>ROC Curve: roc_curve.png</li>
                    <li>Precision-Recall Curve: precision_recall_curve.png</li>
                    <li>Prediction Distribution: prediction_distribution.png</li>
                    <li>Timeline Analysis: timeline_analysis.png</li>
                    <li>Detailed Metrics: evaluation_metrics.json</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Detailed report saved to: {report_path}")
        return report_path
    
    def _rate_metric(self, value: float, good_threshold: float, fair_threshold: float) -> str:
        """Rate a metric as Good/Fair/Poor"""
        if value >= good_threshold:
            return "Good"
        elif value >= fair_threshold:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_strengths(self, metrics: Dict) -> str:
        """Generate list of model strengths"""
        strengths = []
        
        if metrics['accuracy'] > 0.6:
            strengths.append("<li>Good overall accuracy (>60%)</li>")
        if metrics['roc_auc'] > 0.65:
            strengths.append("<li>Strong discriminative ability (ROC-AUC >0.65)</li>")
        if metrics['precision'] > 0.6:
            strengths.append("<li>High precision in up-day predictions</li>")
        if metrics['recall'] > 0.6:
            strengths.append("<li>Good recall for capturing up days</li>")
        if abs(metrics['up_day_accuracy'] - metrics['down_day_accuracy']) < 0.1:
            strengths.append("<li>Balanced performance across market conditions</li>")
        
        if not strengths:
            strengths.append("<li>Model shows potential but requires improvement</li>")
        
        return "\n                    ".join(strengths)
    
    def _generate_improvements(self, metrics: Dict) -> str:
        """Generate list of areas for improvement"""
        improvements = []
        
        if metrics['accuracy'] < 0.55:
            improvements.append("<li>Overall accuracy needs improvement (currently below 55%)</li>")
        if metrics['roc_auc'] < 0.55:
            improvements.append("<li>Model shows limited discriminative ability</li>")
        if metrics['precision'] < 0.5:
            improvements.append("<li>High false positive rate in up-day predictions</li>")
        if metrics['recall'] < 0.5:
            improvements.append("<li>Missing too many actual up days</li>")
        if abs(metrics['up_day_accuracy'] - metrics['down_day_accuracy']) > 0.2:
            improvements.append("<li>Imbalanced performance across market conditions</li>")
        
        if not improvements:
            improvements.append("<li>Continue hyperparameter tuning for optimal performance</li>")
        
        return "\n                    ".join(improvements)
    
    def _generate_trading_insights(self, metrics: Dict, high_conf_accuracy: float) -> str:
        """Generate trading-specific insights"""
        insights = []
        
        if high_conf_accuracy > 0.7:
            insights.append("<li>High-confidence predictions show strong reliability for trading decisions</li>")
        if metrics['up_day_accuracy'] > metrics['down_day_accuracy']:
            insights.append("<li>Model is better at predicting market uptrends</li>")
        elif metrics['down_day_accuracy'] > metrics['up_day_accuracy']:
            insights.append("<li>Model is better at predicting market downtrends</li>")
        
        if metrics['precision'] > 0.6:
            insights.append("<li>Low false positive rate makes it suitable for long positions</li>")
        if metrics['recall'] > 0.6:
            insights.append("<li>Good at capturing most upward movements</li>")
        
        insights.append("<li>Consider using probability thresholds other than 0.5 for trading decisions</li>")
        insights.append("<li>Combine with other indicators for improved trading performance</li>")
        
        return "\n                    ".join(insights)
    
    def run_evaluation(self, output_dir: str = None) -> Dict[str, Any]:
        """Run complete model evaluation"""
        if output_dir is None:
            output_dir = f"/app/experiments/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🔍 Starting comprehensive model evaluation")
        logger.info(f"Output directory: {output_dir}")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Create visualizations
        viz_paths = self.create_visualizations(output_dir)
        
        # Generate detailed report
        report_path = self.generate_detailed_report(metrics, output_dir)
        
        # Save metrics to JSON
        metrics_path = f"{output_dir}/evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'date': self.dates_test,
            'actual': self.y_test,
            'predicted': self.predictions,
            'probability': self.probabilities,
            'correct': self.predictions == self.y_test
        })
        
        predictions_path = f"{output_dir}/predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        
        results = {
            'metrics': metrics,
            'visualizations': viz_paths,
            'report_path': report_path,
            'metrics_path': metrics_path,
            'predictions_path': predictions_path,
            'output_dir': output_dir
        }
        
        logger.info("✅ Model evaluation completed successfully!")
        logger.info(f"📊 Key Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        return results


def load_evaluation_config(config_path: str = None) -> Dict[str, Any]:
    """Load evaluation configuration"""
    if config_path is None:
        config_path = "/app/configs/evaluation_config.json"
    
    # Default configuration
    default_config = {
        "data_split": {
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1
        },
        "visualization": {
            "style": "seaborn-v0_8",
            "dpi": 300,
            "figsize": [10, 6]
        },
        "thresholds": {
            "confidence_threshold": 0.7,
            "good_accuracy": 0.6,
            "fair_accuracy": 0.55
        }
    }
    
    # Try to load custom config
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            # Merge configs
            default_config.update(custom_config)
            logger.info(f"Configuration loaded from: {config_path}")
        else:
            logger.info("Using default evaluation configuration")
    
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default evaluation configuration")
    
    return default_config


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='SENSEX ConvLSTM Model Evaluation')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to the trained model file')
    parser.add_argument('--config', type=str, help='Path to evaluation configuration file')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory for evaluation results')
    parser.add_argument('--log-mlflow', action='store_true', 
                       help='Log results to MLflow')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Load configuration
    config = load_evaluation_config(args.config)
    
    logger.info("🔍 Starting SENSEX ConvLSTM Model Evaluation")
    logger.info(f"Model: {args.model_path}")
    
    try:
        # Initialize evaluator
        evaluator = SensexModelEvaluator(args.model_path, config)
        
        # Run evaluation
        results = evaluator.run_evaluation(args.output_dir)
        
        # Log to MLflow if requested
        if args.log_mlflow:
            logger.info("Logging results to MLflow...")
            
            with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log metrics
                mlflow.log_metrics(results['metrics'])
                
                # Log artifacts
                mlflow.log_artifacts(results['output_dir'], "evaluation_results")
                
                # Log model path as parameter
                mlflow.log_param('evaluated_model_path', args.model_path)
                
                logger.info("Results logged to MLflow")
        
        logger.info("🎉 Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
