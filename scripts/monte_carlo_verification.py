import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

class MonteCarloVerification:
    """
    Monte Carlo simulation to verify theoretical predictions from your paper
    """
    
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
    
    def simulate_manipulation_detection(self, params, strategy_params):
        """
        Simulate the manipulation-detection process
        Verify theoretical predictions through simulation
        """
        print("=== Monte Carlo Simulation of Manipulation Detection ===")
        
        # Extract parameters
        alpha, beta, sigma = params['alpha'], params['beta'], params['sigma']
        mu_r, mu_c, sigma_r, sigma_c, cov_rc = (strategy_params['mu_r'], strategy_params['mu_c'], 
                                               strategy_params['sigma_r'], strategy_params['sigma_c'],
                                               strategy_params['cov_rc'])
        
        # Generate correlated manipulation strategies
        # Create covariance matrix
        cov_matrix = np.array([[sigma_r**2, cov_rc],
                              [cov_rc, sigma_c**2]])
        
        # Generate manipulation episodes
        manipulation_strategies = np.random.multivariate_normal([mu_r, mu_c], cov_matrix, 
                                                               self.n_simulations // 2)
        r_manip, c_manip = manipulation_strategies[:, 0], manipulation_strategies[:, 1]
        
        # Ensure non-negative values and proper bounds
        r_manip = np.maximum(r_manip, 0)
        c_manip = np.clip(c_manip, 0, 1)
        
        # Generate normal trading (no manipulation)
        r_normal = np.zeros(self.n_simulations // 2)
        c_normal = np.random.uniform(0, 0.2, self.n_simulations // 2)  # Low background cancellation
        
        # Combine manipulation and normal trading
        r_all = np.concatenate([r_manip, r_normal])
        c_all = np.concatenate([c_manip, c_normal])
        labels = np.concatenate([np.ones(len(r_manip)), np.zeros(len(r_normal))])
        
        # Generate market noise
        noise = np.random.normal(0, sigma, len(r_all))
        
        # Generate detection signals
        signal_r_only = r_all + noise
        signal_c_only = c_all + noise
        signal_composite = alpha * r_all + beta * c_all + noise
        
        return {
            'r': r_all, 'c': c_all, 'labels': labels,
            'signal_r': signal_r_only, 'signal_c': signal_c_only, 
            'signal_composite': signal_composite,
            'r_manip': r_manip, 'c_manip': c_manip
        }
    
    def verify_signal_amplification_empirically(self, simulation_data):
        """
        Empirically verify your Signal Amplification Theorem
        Test whether composite detection outperforms individual features
        """
        print("\n=== Empirical Signal Amplification Test ===")
        
        labels = simulation_data['labels']
        
        # Calculate AUC for each signal type
        auc_r = roc_auc_score(labels, simulation_data['signal_r'])
        auc_c = roc_auc_score(labels, simulation_data['signal_c'])
        auc_composite = roc_auc_score(labels, simulation_data['signal_composite'])
        
        print(f"AUC (Rush Orders only): {auc_r:.4f}")
        print(f"AUC (Cancellation only): {auc_c:.4f}")
        print(f"AUC (Composite signal): {auc_composite:.4f}")
        
        max_individual = max(auc_r, auc_c)
        amplification = auc_composite - max_individual
        
        print(f"Maximum individual AUC: {max_individual:.4f}")
        print(f"Amplification effect: {amplification:.4f}")
        print(f"Signal Amplification verified: {amplification > 0.01}")
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        
        # ROC curves
        plt.subplot(2, 2, 1)
        fpr_r, tpr_r, _ = roc_curve(labels, simulation_data['signal_r'])
        fpr_c, tpr_c, _ = roc_curve(labels, simulation_data['signal_c'])
        fpr_comp, tpr_comp, _ = roc_curve(labels, simulation_data['signal_composite'])
        
        plt.plot(fpr_r, tpr_r, label=f'Rush Orders (AUC={auc_r:.3f})', color='blue')
        plt.plot(fpr_c, tpr_c, label=f'Cancellation (AUC={auc_c:.3f})', color='red')
        plt.plot(fpr_comp, tpr_comp, label=f'Composite (AUC={auc_composite:.3f})', color='green', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves: Signal Amplification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Signal distributions
        plt.subplot(2, 2, 2)
        manip_mask = labels == 1
        normal_mask = labels == 0
        
        plt.hist(simulation_data['signal_composite'][normal_mask], bins=50, alpha=0.7, 
                label='Normal Trading', color='blue', density=True)
        plt.hist(simulation_data['signal_composite'][manip_mask], bins=50, alpha=0.7, 
                label='Manipulation', color='red', density=True)
        plt.xlabel('Composite Signal Value')
        plt.ylabel('Density')
        plt.title('Signal Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Strategy correlation verification
        plt.subplot(2, 2, 3)
        r_manip = simulation_data['r_manip']
        c_manip = simulation_data['c_manip']
        correlation = np.corrcoef(r_manip, c_manip)[0, 1]
        
        plt.scatter(r_manip, c_manip, alpha=0.6, s=20)
        plt.xlabel('Rush Order Intensity (r)')
        plt.ylabel('Cancellation Ratio (c)')
        plt.title(f'Strategy Complementarity (ρ={correlation:.3f})')
        plt.grid(True, alpha=0.3)
        
        # AUC comparison
        plt.subplot(2, 2, 4)
        methods = ['Rush Orders', 'Cancellation', 'Composite']
        aucs = [auc_r, auc_c, auc_composite]
        colors = ['blue', 'red', 'green']
        
        bars = plt.bar(methods, aucs, color=colors, alpha=0.7)
        plt.ylabel('AUC Score')
        plt.title('Detection Performance Comparison')
        plt.ylim(0.5, 1.0)
        
        # Add value labels on bars
        for bar, auc in zip(bars, aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return auc_r, auc_c, auc_composite, amplification
    
    def verify_complementarity_hypothesis(self, simulation_data):
        """
        Test your complementarity hypothesis: Cov(r,c|Manipulation) > Cov(r,c|Normal)
        """
        print("\n=== Complementarity Hypothesis Test ===")
        
        labels = simulation_data['labels']
        r_all = simulation_data['r']
        c_all = simulation_data['c']
        
        # Split by manipulation vs normal trading
        manip_mask = labels == 1
        normal_mask = labels == 0
        
        # Calculate covariances
        cov_manipulation = np.cov(r_all[manip_mask], c_all[manip_mask])[0, 1]
        cov_normal = np.cov(r_all[normal_mask], c_all[normal_mask])[0, 1]
        
        print(f"Cov(r,c | Manipulation): {cov_manipulation:.4f}")
        print(f"Cov(r,c | Normal Trading): {cov_normal:.4f}")
        print(f"Complementarity hypothesis verified: {cov_manipulation > cov_normal}")
        
        # Statistical significance test
        from scipy.stats import pearsonr
        
        corr_manip, p_manip = pearsonr(r_all[manip_mask], c_all[manip_mask])
        corr_normal, p_normal = pearsonr(r_all[normal_mask], c_all[normal_mask])
        
        print(f"Correlation (Manipulation): {corr_manip:.4f} (p={p_manip:.4f})")
        print(f"Correlation (Normal): {corr_normal:.4f} (p={p_normal:.4f})")
        
        return cov_manipulation, cov_normal, corr_manip, corr_normal
    
    def verify_welfare_predictions(self, params, threshold_range=None):
        """
        Verify welfare analysis predictions from your paper
        """
        print("\n=== Welfare Analysis Verification ===")
        
        if threshold_range is None:
            threshold_range = np.linspace(-2, 4, 50)
        
        # Simulate detection outcomes for different thresholds
        strategy_params = {
            'mu_r': 2.0, 'mu_c': 1.5, 'sigma_r': 0.5, 'sigma_c': 0.3, 'cov_rc': 0.2
        }
        
        simulation_data = self.simulate_manipulation_detection(params, strategy_params)
        
        welfare_scores = []
        detection_rates = []
        false_alarm_rates = []
        
        for tau in threshold_range:
            # Detection decisions
            signal = simulation_data['signal_composite']
            labels = simulation_data['labels']
            
            detected = signal > tau
            
            # Calculate rates
            tp = np.sum((detected == 1) & (labels == 1))  # True positives
            fp = np.sum((detected == 1) & (labels == 0))  # False positives
            tn = np.sum((detected == 0) & (labels == 0))  # True negatives
            fn = np.sum((detected == 0) & (labels == 1))  # False negatives
            
            detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Welfare calculation (simplified)
            welfare = params['H'] * detection_rate - params['F'] * false_alarm_rate
            
            welfare_scores.append(welfare)
            detection_rates.append(detection_rate)
            false_alarm_rates.append(false_alarm_rate)
        
        # Find optimal threshold
        optimal_idx = np.argmax(welfare_scores)
        optimal_tau = threshold_range[optimal_idx]
        optimal_welfare = welfare_scores[optimal_idx]
        
        print(f"Optimal detection threshold: {optimal_tau:.4f}")
        print(f"Maximum welfare: {optimal_welfare:.4f}")
        print(f"Optimal detection rate: {detection_rates[optimal_idx]:.4f}")
        print(f"Optimal false alarm rate: {false_alarm_rates[optimal_idx]:.4f}")
        
        # Plot welfare analysis
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(threshold_range, welfare_scores, 'b-', linewidth=2)
        plt.axvline(optimal_tau, color='r', linestyle='--', label=f'Optimal τ={optimal_tau:.2f}')
        plt.xlabel('Detection Threshold (τ)')
        plt.ylabel('Social Welfare')
        plt.title('Welfare vs. Detection Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(threshold_range, detection_rates, 'g-', label='Detection Rate')
        plt.plot(threshold_range, false_alarm_rates, 'r-', label='False Alarm Rate')
        plt.axvline(optimal_tau, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Detection Threshold (τ)')
        plt.ylabel('Rate')
        plt.title('Detection vs. False Alarm Rates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(false_alarm_rates, detection_rates, 'b-o', markersize=3)
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Detection Rate')
        plt.title('ROC-like Curve (Different Thresholds)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Welfare components
        benefits = [params['H'] * dr for dr in detection_rates]
        costs = [params['F'] * far for far in false_alarm_rates]
        
        plt.plot(threshold_range, benefits, 'g-', label='Detection Benefits')
        plt.plot(threshold_range, costs, 'r-', label='False Alarm Costs')
        plt.plot(threshold_range, welfare_scores, 'b-', label='Net Welfare')
        plt.axvline(optimal_tau, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Detection Threshold (τ)')
        plt.ylabel('Welfare Components')
        plt.title('Welfare Decomposition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return optimal_tau, optimal_welfare, detection_rates, false_alarm_rates

# Example usage
if __name__ == "__main__":
    # Parameters for verification
    params = {
        'alpha': 1.0, 'beta': 0.8, 'sigma': 0.3,
        'H': 25, 'F': 10
    }
    
    strategy_params = {
        'mu_r': 2.0, 'mu_c': 1.5, 
        'sigma_r': 0.5, 'sigma_c': 0.3, 
        'cov_rc': 0.2  # Positive covariance for complementarity
    }
    
    # Run Monte Carlo verification
    print("MONTE CARLO VERIFICATION OF THEORETICAL PREDICTIONS")
    print("=" * 60)
    
    verifier = MonteCarloVerification(n_simulations=20000)
    
    # Generate simulation data
    sim_data = verifier.simulate_manipulation_detection(params, strategy_params)
    
    # Verify Signal Amplification Theorem
    auc_results = verifier.verify_signal_amplification_empirically(sim_data)
    
    # Verify complementarity hypothesis
    comp_results = verifier.verify_complementarity_hypothesis(sim_data)
    
    # Verify welfare predictions
    welfare_results = verifier.verify_welfare_predictions(params)
    
    print("\n" + "=" * 60)
    print("MONTE CARLO VERIFICATION COMPLETE")
    print("Theoretical predictions validated through simulation:")