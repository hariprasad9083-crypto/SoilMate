import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report, r2_score, 
                           mean_absolute_error, mean_squared_error)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import SoilMatePreprocessor

class ModelComparison:
    def __init__(self):
        self.classification_models = {}
        self.regression_models = {}
        self.results = {}
        
    def initialize_classification_models(self):
        """Initialize classification models for crop recommendation"""
        self.classification_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=15
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                multi_class='ovr'
            ),
            'SVM': SVC(
                kernel='rbf', 
                random_state=42,
                probability=True
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=6
            )
        }
    
    def initialize_regression_models(self):
        """Initialize regression models for economic predictions"""
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=15
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=6
            )
        }
    
    def evaluate_classification_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all classification models"""
        self.initialize_classification_models()
        results = {}
        
        for name, model in self.classification_models.items():
            print(f"Training {name}...")
            
            # Training time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prediction time
            start_time = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            prediction_time = time.time() - start_time
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                      scoring='accuracy')
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'prediction_time': prediction_time,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        self.classification_results = results
        return results
    
    def evaluate_regression_models(self, X_train, X_test, y_train, y_test, target_name):
        """Evaluate regression models for economic predictions"""
        self.initialize_regression_models()
        results = {}
        
        for name, model in self.regression_models.items():
            print(f"Training {name} for {target_name}...")
            
            # Training time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prediction time
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                      scoring='r2')
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'model': model,
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'prediction_time': prediction_time,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            print(f"{name} - R²: {r2:.4f}, MAE: {mae:.2f}")
        
        return results
    
    def create_classification_comparison_plots(self, results, save_dir='plots/'):
        """Create comprehensive comparison plots for classification"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx//2, idx%2]
            values = [results[model][metric] for model in models]
            cv_stds = [results[model]['cv_std'] for model in models]
            
            bars = ax.bar(models, values, alpha=0.8, capsize=5)
            ax.errorbar(models, values, yerr=cv_stds, fmt='none', color='black', capsize=5)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}classification_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training time vs accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        
        accuracies = [results[model]['accuracy'] for model in models]
        times = [results[model]['training_time'] for model in models]
        
        scatter = ax.scatter(times, accuracies, s=200, alpha=0.7)
        
        for i, model in enumerate(models):
            ax.annotate(model, (times[i], accuracies[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Time vs Accuracy')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f'{save_dir}time_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cross-validation score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cv_data = []
        for model in models:
            # Simulate CV scores for visualization (since we only have mean/std)
            cv_mean = results[model]['cv_mean']
            cv_std = results[model]['cv_std']
            cv_scores = np.random.normal(cv_mean, cv_std, 5)
            cv_data.append(cv_scores)
        
        ax.boxplot(cv_data, labels=models)
        ax.set_title('Cross-Validation Score Distribution')
        ax.set_ylabel('Accuracy Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}cv_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Classification plots saved to {save_dir}")
    
    def create_regression_comparison_plots(self, results, target_name, save_dir='plots/'):
        """Create comparison plots for regression models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Performance comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = list(results.keys())
        metrics = ['r2_score', 'mae', 'rmse']
        metric_names = ['R² Score', 'MAE', 'RMSE']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            values = [results[model][metric] for model in models]
            
            bars = ax.bar(models, values, alpha=0.8)
            ax.set_title(f'{metric_name} Comparison - {target_name}')
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       f'{value:.3f}' if metric == 'r2_score' else f'{value:.0f}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}regression_{target_name}_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Actual vs Predicted plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, model in enumerate(models):
            ax = axes[idx//2, idx%2]
            y_test = results[model]['y_test']
            y_pred = results[model]['y_pred']
            
            ax.scatter(y_test, y_pred, alpha=0.6)
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model} - Actual vs Predicted')
            
            # R² score on plot
            r2 = results[model]['r2_score']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}regression_{target_name}_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Regression plots for {target_name} saved to {save_dir}")
    
    def select_best_models(self, classification_results, economic_results):
        """Select best models based on performance criteria"""
        
        # Best classification model (weighted criteria)
        best_classification = None
        best_classification_score = 0
        
        for name, result in classification_results.items():
            # Weighted score: 40% accuracy, 30% f1, 20% speed, 10% interpretability
            speed_score = 1 / (1 + result['training_time'])  # Inverse time score
            interpretability_score = 1.0 if 'Forest' in name else 0.5  # Tree models more interpretable
            
            combined_score = (0.4 * result['accuracy'] + 
                            0.3 * result['f1_score'] + 
                            0.2 * speed_score + 
                            0.1 * interpretability_score)
            
            if combined_score > best_classification_score:
                best_classification_score = combined_score
                best_classification = name
        
        # Best economic models
        best_economic = {}
        for target, results in economic_results.items():
            best_model = None
            best_score = -np.inf
            
            for name, result in results.items():
                # Weighted score: 50% R², 30% MAE (inverted), 20% speed
                speed_score = 1 / (1 + result['training_time'])
                mae_score = 1 / (1 + result['mae'] / 1000)  # Normalize MAE
                
                combined_score = (0.5 * result['r2_score'] + 
                                0.3 * mae_score + 
                                0.2 * speed_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_model = name
            
            best_economic[target] = best_model
        
        return best_classification, best_economic
    
    def save_best_models(self, classification_results, economic_results, 
                        best_classification, best_economic, save_dir='models/'):
        """Save the best performing models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best classification model
        best_clf_model = classification_results[best_classification]['model']
        joblib.dump(best_clf_model, f'{save_dir}best_crop_classifier.pkl')
        
        # Save best economic models
        for target, model_name in best_economic.items():
            best_model = economic_results[target][model_name]['model']
            joblib.dump(best_model, f'{save_dir}best_{target}_model.pkl')
        
        # Save selection report
        selection_report = {
            'best_classification': best_classification,
            'best_economic': best_economic,
            'classification_performance': {
                name: {k: v for k, v in result.items() 
                      if k not in ['model', 'y_pred', 'y_pred_proba', 'confusion_matrix']}
                for name, result in classification_results.items()
            }
        }
        
        import json
        with open(f'{save_dir}model_selection_report.json', 'w') as f:
            json.dump(selection_report, f, indent=2, default=str)
        
        print(f"Best models saved to {save_dir}")
        print(f"Best classification model: {best_classification}")
        print(f"Best economic models: {best_economic}")

def main():
    """Run complete model comparison"""
    
    # Initialize components
    preprocessor = SoilMatePreprocessor()
    comparison = ModelComparison()
    
    # Load and prepare data
    preprocessor.load_data('Final_SoilMate_Dataset.csv', 'Final_SoilMate_Crop_Prices.csv')
    
    print("=== CROP CLASSIFICATION MODEL COMPARISON ===")
    
    # Prepare crop classification data
    X_crop, y_crop = preprocessor.prepare_crop_classification_data()
    X_train_crop, X_test_crop, y_train_crop, y_test_crop = preprocessor.create_train_test_split(X_crop, y_crop)
    X_train_crop_scaled, X_test_crop_scaled = preprocessor.scale_features(X_train_crop, X_test_crop, scaler_type='crop')
    
    # Evaluate classification models
    classification_results = comparison.evaluate_classification_models(
        X_train_crop_scaled, X_test_crop_scaled, y_train_crop, y_test_crop
    )
    
    # Create classification plots
    comparison.create_classification_comparison_plots(classification_results)
    
    print("\n=== ECONOMIC MODEL COMPARISON ===")
    
    # Evaluate economic models
    economic_results = {}
    economic_targets = ['cost', 'revenue', 'profit']
    
    for target in economic_targets:
        print(f"\nEvaluating {target} models...")
        X_econ, y_econ = preprocessor.prepare_economic_data(target)
        X_train_econ, X_test_econ, y_train_econ, y_test_econ = train_test_split(
            X_econ, y_econ, test_size=0.2, random_state=42
        )
        X_train_econ_scaled, X_test_econ_scaled = preprocessor.scale_features(
            X_train_econ, X_test_econ, fit_scaler=(target == 'cost'), scaler_type='economic'
        )
        
        results = comparison.evaluate_regression_models(
            X_train_econ_scaled, X_test_econ_scaled, y_train_econ, y_test_econ, target
        )
        economic_results[target] = results
        
        # Create plots for this target
        comparison.create_regression_comparison_plots(results, target)
    
    # Select and save best models
    best_classification, best_economic = comparison.select_best_models(
        classification_results, economic_results
    )
    
    comparison.save_best_models(
        classification_results, economic_results, 
        best_classification, best_economic
    )
    
    # Save preprocessors
    preprocessor.save_preprocessors()
    
    print("\n=== MODEL COMPARISON COMPLETE ===")
    return comparison, classification_results, economic_results

if __name__ == "__main__":
    comparison, classification_results, economic_results = main()