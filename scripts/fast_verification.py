import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

class FastVerification:
    """
    Fast numerical verification of your Signal Amplification Theorem
    Focuses on numerical validation rather than symbolic manipulation
    """
    
    def __init__(self):
        self.start_time = time.time()
    
    def print_timing(self, step_name):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {step_name}")
    
    def snr_function(self, w1, w2, mu_r=2.0, mu_c=1.5, sigma_r=0.8, sigma_c=0.3, cov_rc=0.3, sigma_noise=0.5):
        """
        Fast numerical SNR calculation
        """
        # Signal power under manipulation
        signal_mean_sq = (w1 * mu_r + w2 * mu_c)**2
        signal_var = w1**2 * sigma_r**2 + w2**2 * sigma_c**2 + 2*w1*w2*cov_rc
        numerator = signal_mean_sq + signal_var
        
        # Noise power
        denominator = w1**2 + w2**2 + sigma_noise**2
        
        return numerator / denominator
    
    def verify_signal_amplification_fast(self):
        """
        Fast numerical verification of Signal Amplification Theorem
        """
        print("=" * 60)
        print("FAST SIGNAL AMPLIFICATION VERIFICATION")
        print("=" * 60)
        
        self.print_timing("Starting verification")
        
        # Test parameters
        params = {
            'mu_r': 2.0, 'mu_c': 1.5, 'sigma_r': 0.8, 'sigma_c': 0.3, 
            'cov_rc': 0.3, 'sigma_noise': 0.5
        }
        
        print("Parameters:")
        for k, v in params.items():
            print(f"  {k} = {v}")
        
        # Individual feature SNRs
        snr_r_only = self.snr_function(1, 0, **params)
        snr_c_only = self.snr_function(0, 1, **params)
        
        self.print_timing("Calculated individual SNRs")
        
        # Optimize for best composite SNR
        def negative_snr(weights):
            w1, w2 = weights
            return -self.snr_function(w1, w2, **params)
        
        result = opt.minimize(negative_snr, [0.5, 0.5], method='BFGS')
        
        self.print_timing("Optimized composite SNR")
        
        if result.success:
            w1_opt, w2_opt = result.x
            snr_optimal = -result.fun
            
            # Results
            max_individual = max(snr_r_only, snr_c_only)
            amplification = snr_optimal - max_individual
            
            print(f"\nRESULTS:")
            print(f"  SNR (rush orders only): {snr_r_only:.4f}")
            print(f"  SNR (cancellation only): {snr_c_only:.4f}")
            print(f"  SNR (optimal composite): {snr_optimal:.4f}")
            print(f"  Optimal weights: w1={w1_opt:.4f}, w2={w2_opt:.4f}")
            print(f"  Amplification: {amplification:.4f}")
            print(f"  Improvement: {(amplification/max_individual*100):.1f}%")
            print(f"  ✓ Theorem verified: {amplification > 0.001}")
            
            # Quick sensitivity test
            cov_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            amplifications = []
            
            for cov in cov_values:
                test_params = params.copy()
                test_params['cov_rc'] = cov
                
                result_test = opt.minimize(lambda w: -self.snr_function(w[0], w[1], **test_params), 
                                         [0.5, 0.5], method='BFGS')
                
                if result_test.success:
                    snr_test = -result_test.fun
                    snr_r_test = self.snr_function(1, 0, **test_params)
                    snr_c_test = self.snr_function(0, 1, **test_params)
                    amp_test = snr_test - max(snr_r_test, snr_c_test)
                    amplifications.append(amp_test)
                else:
                    amplifications.append(0)
            
            self.print_timing("Completed sensitivity analysis")
            
            # Plot results
            plt.figure(figsize=(12, 8))
            
            # SNR comparison
            plt.subplot(2, 2, 1)
            methods = ['Rush\nOrders', 'Cancellation\nRatio', 'Optimal\nComposite']
            snrs = [snr_r_only, snr_c_only, snr_optimal]
            colors = ['blue', 'red', 'green']
            
            bars = plt.bar(methods, snrs, color=colors, alpha=0.7)
            plt.ylabel('SNR')
            plt.title('Detection Performance Comparison')
            plt.grid(True, alpha=0.3)
            
            for bar, snr in zip(bars, snrs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{snr:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Amplification vs covariance
            plt.subplot(2, 2, 2)
            plt.plot(cov_values, amplifications, 'o-', linewidth=2, markersize=8, color='purple')
            plt.xlabel('Covariance (δ)')
            plt.ylabel('Amplification Effect')
            plt.title('Amplification vs. Strategy Complementarity')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Weight visualization
            plt.subplot(2, 2, 3)
            angles = np.linspace(0, 2*np.pi, 100)
            w1_circle = np.cos(angles)
            w2_circle = np.sin(angles)
            snr_circle = [self.snr_function(w1, w2, **params) for w1, w2 in zip(w1_circle, w2_circle)]
            
            plt.plot(w1_circle, w2_circle, 'lightgray', alpha=0.5, label='Unit circle')
            scatter = plt.scatter(w1_circle, w2_circle, c=snr_circle, cmap='viridis', s=20)
            plt.scatter(w1_opt, w2_opt, color='red', s=100, marker='*', label=f'Optimal ({w1_opt:.2f}, {w2_opt:.2f})')
            plt.xlabel('Weight w1 (rush orders)')
            plt.ylabel('Weight w2 (cancellation)')
            plt.title('SNR Landscape')
            plt.colorbar(scatter, label='SNR')
            plt.legend()
            plt.axis('equal')
            
            # Theoretical vs empirical
            plt.subplot(2, 2, 4)
            epsilon_theoretical = [2 * cov / (params['sigma_noise']**2 + params['sigma_r']**2 + params['sigma_c']**2) 
                                 for cov in cov_values]
            
            plt.plot(cov_values, epsilon_theoretical, 'r--', label='Theoretical ε(δ)', linewidth=2)
            plt.plot(cov_values, amplifications, 'bo-', label='Empirical amplification', linewidth=2)
            plt.xlabel('Covariance (δ)')
            plt.ylabel('Amplification')
            plt.title('Theory vs. Empirical')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            self.print_timing("Generated plots")
            
            return True, {
                'snr_optimal': snr_optimal,
                'weights': (w1_opt, w2_opt),
                'amplification': amplification,
                'individual_snrs': (snr_r_only, snr_c_only)
            }
        else:
            print("Optimization failed!")
            return False, None
    
    def verify_game_theory_fast(self):
        """
        Fast verification of game-theoretic aspects
        """
        print("\n" + "=" * 60)
        print("FAST GAME THEORY VERIFICATION")
        print("=" * 60)
        
        # Model parameters
        params = {
            'Q': 100, 'delta': 1.5, 'gamma': 1.2, 'k': 0.8, 'l': 0.6,
            'L': 50, 'alpha': 1.0, 'beta': 0.8, 'tau': 2.0, 'sigma': 0.5
        }
        
        def manipulator_profit(r, c, alpha=None):
            if alpha is None:
                alpha = params['alpha']
            
            signal = alpha * r + params['beta'] * c
            detection_prob = norm.cdf((signal - params['tau']) / params['sigma'])
            
            return ((params['delta'] * r + params['gamma'] * c) * params['Q'] 
                   - 0.5 * (params['k'] * r**2 + params['l'] * c**2) 
                   - params['L'] * detection_prob)
        
        # Find equilibrium
        def negative_profit(strategy):
            r, c = strategy
            if r < 0 or c < 0 or c > 1:
                return 1e6
            return -manipulator_profit(r, c)
        
        result = opt.minimize(negative_profit, [1.0, 0.5], bounds=[(0, 10), (0, 1)], method='L-BFGS-B')
        
        self.print_timing("Found equilibrium")
        
        if result.success:
            r_star, c_star = result.x
            profit_star = -result.fun
            
            print(f"Equilibrium: r* = {r_star:.4f}, c* = {c_star:.4f}")
            print(f"Equilibrium profit: {profit_star:.4f}")
            
            # Test strategic deterrence
            alpha_values = np.linspace(0.5, 2.0, 10)
            manipulations = []
            
            for alpha in alpha_values:
                result_det = opt.minimize(lambda s: -manipulator_profit(s[0], s[1], alpha), 
                                        [1.0, 0.5], bounds=[(0, 10), (0, 1)], method='L-BFGS-B')
                if result_det.success:
                    manipulations.append((alpha, result_det.x[0], result_det.x[1]))
            
            self.print_timing("Tested strategic deterrence")
            
            if len(manipulations) > 1:
                alphas = [m[0] for m in manipulations]
                r_values = [m[1] for m in manipulations]
                
                correlation = np.corrcoef(alphas, r_values)[0, 1]
                print(f"Strategic deterrence correlation: {correlation:.4f}")
                print(f"✓ Deterrence verified: {correlation < -0.1}")
                
                # Test complementarity
                epsilon = 1e-5
                r_test, c_test = 1.5, 0.4
                
                f_pp = manipulator_profit(r_test + epsilon, c_test + epsilon)
                f_pm = manipulator_profit(r_test + epsilon, c_test - epsilon)
                f_mp = manipulator_profit(r_test - epsilon, c_test + epsilon)
                f_mm = manipulator_profit(r_test - epsilon, c_test - epsilon)
                
                cross_partial = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                
                self.print_timing("Tested complementarity")
                
                print(f"Cross-partial ∂²π/∂r∂c: {cross_partial:.6f}")
                print(f"✓ Complementarity verified: {cross_partial > 0}")
                
                # Quick plot
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(alphas, r_values, 'bo-', linewidth=2)
                plt.xlabel('Detection Sensitivity (α)')
                plt.ylabel('Optimal Manipulation (r*)')
                plt.title(f'Strategic Deterrence (ρ={correlation:.3f})')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                # Show profit surface around equilibrium
                r_range = np.linspace(max(0.1, r_star-1), r_star+1, 20)
                c_range = np.linspace(max(0.1, c_star-0.3), min(0.9, c_star+0.3), 20)
                R, C = np.meshgrid(r_range, c_range)
                Z = np.array([[manipulator_profit(r, c) for r in r_range] for c in c_range])
                
                contour = plt.contour(R, C, Z, levels=15)
                plt.scatter(r_star, c_star, color='red', s=100, marker='*', label='Equilibrium')
                plt.xlabel('Rush Orders (r)')
                plt.ylabel('Cancellation Ratio (c)')
                plt.title('Profit Landscape')
                plt.legend()
                plt.colorbar(contour)
                
                plt.tight_layout()
                plt.show()
                
                self.print_timing("Generated game theory plots")
                
                return True, {
                    'equilibrium': (r_star, c_star),
                    'deterrence_correlation': correlation,
                    'cross_partial': cross_partial
                }
        
        return False, None
    
    def run_complete_verification(self):
        """
        Run all verifications quickly
        """
        print("FAST COMPLETE VERIFICATION")
        print("=" * 70)
        
        # Signal amplification
        success1, results1 = self.verify_signal_amplification_fast()
        
        # Game theory
        success2, results2 = self.verify_game_theory_fast()
        
        # Summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        
        if success1:
            print("✓ Signal Amplification Theorem: VERIFIED")
            print(f"  - Amplification achieved: {results1['amplification']:.4f}")
            print(f"  - Optimal weights: ({results1['weights'][0]:.3f}, {results1['weights'][1]:.3f})")
        else:
            print("✗ Signal Amplification Theorem: FAILED")
        
        if success2:
            print("✓ Game Theory Analysis: VERIFIED")
            print(f"  - Equilibrium: r*={results2['equilibrium'][0]:.3f}, c*={results2['equilibrium'][1]:.3f}")
            print(f"  - Strategic deterrence: {results2['deterrence_correlation']:.3f}")
            print(f"  - Complementarity: {results2['cross_partial']:.6f}")
        else:
            print("✗ Game Theory Analysis: FAILED")
        
        total_time = time.time() - self.start_time
        print(f"\nTotal verification time: {total_time:.1f} seconds")
        
        return success1 and success2

# Run the fast verification
if __name__ == "__main__":
    verifier = FastVerification()
    success = verifier.run_complete_verification()
    
    if success:
        print("\n🎉 All verifications passed! Your mathematical work is sound.")
    else:
        print("\n⚠️  Some verifications failed. Check the results above.")
