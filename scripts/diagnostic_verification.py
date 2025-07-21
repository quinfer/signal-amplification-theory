import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
import matplotlib.pyplot as plt

class DiagnosticVerification:
    """
    Diagnostic tool to identify issues with the mathematical model
    """
    
    def __init__(self):
        pass
    
    def analyze_snr_issues(self):
        """
        Diagnose why SNR optimization is failing
        """
        print("=" * 60)
        print("DIAGNOSING SNR OPTIMIZATION ISSUES")
        print("=" * 60)
        
        def snr_function(w1, w2, mu_r=2.0, mu_c=1.5, sigma_r=0.8, sigma_c=0.3, cov_rc=0.3, sigma_noise=0.5):
            # Signal power under manipulation
            signal_mean_sq = (w1 * mu_r + w2 * mu_c)**2
            signal_var = w1**2 * sigma_r**2 + w2**2 * sigma_c**2 + 2*w1*w2*cov_rc
            numerator = signal_mean_sq + signal_var
            
            # Noise power
            denominator = w1**2 + w2**2 + sigma_noise**2
            
            print(f"  w1={w1:.3f}, w2={w2:.3f}")
            print(f"  Signal mean²: {signal_mean_sq:.4f}")
            print(f"  Signal var: {signal_var:.4f}")
            print(f"  Numerator: {numerator:.4f}")
            print(f"  Denominator: {denominator:.4f}")
            
            if denominator <= 0:
                print("  ERROR: Denominator ≤ 0!")
                return 0
            
            snr = numerator / denominator
            print(f"  SNR: {snr:.4f}")
            return snr
        
        # Test individual SNRs
        print("\n1. Testing individual feature SNRs:")
        print("Rush orders only (w1=1, w2=0):")
        snr_r = snr_function(1, 0)
        
        print("\nCancellation only (w1=0, w2=1):")
        snr_c = snr_function(0, 1)
        
        print("\n2. Testing composite SNR with equal weights:")
        print("Equal weights (w1=0.5, w2=0.5):")
        snr_equal = snr_function(0.5, 0.5)
        
        print("\n3. Testing different covariance values:")
        for cov in [0, 0.1, 0.3, 0.5]:
            print(f"\nCovariance = {cov}:")
            snr_test = snr_function(0.5, 0.5, cov_rc=cov)
        
        # Try manual optimization
        print("\n4. Manual weight optimization:")
        best_snr = 0
        best_weights = (0, 0)
        
        for w1 in np.linspace(0, 2, 11):
            for w2 in np.linspace(0, 2, 11):
                try:
                    snr = snr_function(w1, w2)
                    if snr > best_snr:
                        best_snr = snr
                        best_weights = (w1, w2)
                except:
                    continue
        
        print(f"Best SNR found: {best_snr:.4f} at weights {best_weights}")
        
        return best_snr, best_weights
    
    def analyze_game_theory_issues(self):
        """
        Diagnose game theory problems
        """
        print("\n" + "=" * 60)
        print("DIAGNOSING GAME THEORY ISSUES")
        print("=" * 60)
        
        # Check if parameters are reasonable
        params = {
            'Q': 100, 'delta': 1.5, 'gamma': 1.2, 'k': 0.8, 'l': 0.6,
            'L': 50, 'alpha': 1.0, 'beta': 0.8, 'tau': 2.0, 'sigma': 0.5
        }
        
        print("Model parameters:")
        for k, v in params.items():
            print(f"  {k} = {v}")
        
        def manipulator_profit(r, c, alpha=None, verbose=False):
            if alpha is None:
                alpha = params['alpha']
            
            # Check bounds
            if r < 0 or c < 0 or c > 1:
                if verbose:
                    print(f"  Constraint violation: r={r:.3f}, c={c:.3f}")
                return -1e6
            
            signal = alpha * r + params['beta'] * c
            detection_prob = norm.cdf((signal - params['tau']) / params['sigma'])
            
            revenue = (params['delta'] * r + params['gamma'] * c) * params['Q']
            costs = 0.5 * (params['k'] * r**2 + params['l'] * c**2)
            penalty = params['L'] * detection_prob
            
            profit = revenue - costs - penalty
            
            if verbose:
                print(f"  r={r:.3f}, c={c:.3f}")
                print(f"  Signal: {signal:.3f}")
                print(f"  Detection prob: {detection_prob:.4f}")
                print(f"  Revenue: {revenue:.2f}")
                print(f"  Costs: {costs:.2f}")
                print(f"  Penalty: {penalty:.2f}")
                print(f"  Profit: {profit:.2f}")
            
            return profit
        
        print("\n1. Testing profit function at different points:")
        
        test_points = [(0, 0), (1, 0.5), (2, 0.5), (5, 0.5), (10, 1.0)]
        for r, c in test_points:
            print(f"\nPoint (r={r}, c={c}):")
            profit = manipulator_profit(r, c, verbose=True)
        
        print("\n2. Checking why equilibrium is at boundary:")
        
        # Test if profit is always increasing in r and c
        r_test = np.linspace(0.1, 5, 20)
        profits_r = [manipulator_profit(r, 0.5) for r in r_test]
        
        c_test = np.linspace(0.1, 0.9, 20)
        profits_c = [manipulator_profit(2, c) for c in c_test]
        
        print(f"Profit trend with r: {np.diff(profits_r)[:5]} (should decrease)")
        print(f"Profit trend with c: {np.diff(profits_c)[:5]} (should decrease)")
        
        # Check first derivatives numerically
        epsilon = 1e-6
        r0, c0 = 2.0, 0.5
        
        dprofit_dr = (manipulator_profit(r0 + epsilon, c0) - manipulator_profit(r0 - epsilon, c0)) / (2 * epsilon)
        dprofit_dc = (manipulator_profit(r0, c0 + epsilon) - manipulator_profit(r0, c0 - epsilon)) / (2 * epsilon)
        
        print(f"\nFirst derivatives at (r={r0}, c={c0}):")
        print(f"  ∂π/∂r = {dprofit_dr:.6f} (should be ≈ 0 at equilibrium)")
        print(f"  ∂π/∂c = {dprofit_dc:.6f} (should be ≈ 0 at equilibrium)")
        
        # Check cross-partial
        d2profit_drdc = (manipulator_profit(r0 + epsilon, c0 + epsilon) - 
                        manipulator_profit(r0 + epsilon, c0 - epsilon) - 
                        manipulator_profit(r0 - epsilon, c0 + epsilon) + 
                        manipulator_profit(r0 - epsilon, c0 - epsilon)) / (4 * epsilon**2)
        
        print(f"  ∂²π/∂r∂c = {d2profit_drdc:.6f} (should be > 0 for complementarity)")
        
        return dprofit_dr, dprofit_dc, d2profit_drdc
    
    def suggest_parameter_fixes(self):
        """
        Suggest parameter adjustments to fix the issues
        """
        print("\n" + "=" * 60)
        print("SUGGESTED PARAMETER FIXES")
        print("=" * 60)
        
        print("ISSUES IDENTIFIED:")
        print("1. SNR optimization failing → weights might be unbounded")
        print("2. Equilibrium at boundary (r=10, c=1) → costs too low relative to benefits")
        print("3. Negative complementarity → detection penalty not creating right incentives")
        print("4. Strategic deterrence NaN → no variation in optimal strategies")
        
        print("\nSUGGESTED FIXES:")
        
        # Fix 1: SNR denominator issue
        print("\n1. SNR Function:")
        print("   ISSUE: Denominator in SNR may approach zero")
        print("   FIX: Add regularization term to denominator")
        
        def fixed_snr(w1, w2, mu_r=2.0, mu_c=1.5, sigma_r=0.8, sigma_c=0.3, cov_rc=0.3, sigma_noise=0.5):
            signal_mean_sq = (w1 * mu_r + w2 * mu_c)**2
            signal_var = w1**2 * sigma_r**2 + w2**2 * sigma_c**2 + 2*w1*w2*cov_rc
            numerator = signal_mean_sq + signal_var
            
            # Fixed denominator with regularization
            denominator = max(w1**2 + w2**2 + sigma_noise**2, 0.01)
            
            return numerator / denominator
        
        # Test fixed SNR
        try:
            result = opt.minimize(lambda w: -fixed_snr(w[0], w[1]), [0.5, 0.5], method='BFGS')
            if result.success:
                print(f"   FIXED SNR: Optimal SNR = {-result.fun:.4f} at weights {result.x}")
        except:
            print("   FIXED SNR: Still failing")
        
        # Fix 2: Increase costs to create interior equilibrium
        print("\n2. Game Theory Parameters:")
        print("   ISSUE: Costs (k, l) too low → no interior optimum")
        print("   FIX: Increase cost parameters")
        
        suggested_params = {
            'Q': 100, 'delta': 1.5, 'gamma': 1.2, 
            'k': 5.0,     # Increased from 0.8
            'l': 4.0,     # Increased from 0.6
            'L': 50, 'alpha': 1.0, 'beta': 0.8, 'tau': 2.0, 'sigma': 0.5
        }
        
        def test_profit(r, c, params):
            signal = params['alpha'] * r + params['beta'] * c
            detection_prob = norm.cdf((signal - params['tau']) / params['sigma'])
            
            revenue = (params['delta'] * r + params['gamma'] * c) * params['Q']
            costs = 0.5 * (params['k'] * r**2 + params['l'] * c**2)
            penalty = params['L'] * detection_prob
            
            return revenue - costs - penalty
        
        # Test with new parameters
        result = opt.minimize(lambda s: -test_profit(s[0], s[1], suggested_params), 
                            [1.0, 0.5], bounds=[(0, 10), (0, 1)], method='L-BFGS-B')
        
        if result.success:
            r_new, c_new = result.x
            print(f"   FIXED EQUILIBRIUM: r* = {r_new:.4f}, c* = {c_new:.4f}")
            print(f"   (Interior solution: {0 < r_new < 10 and 0 < c_new < 1})")
        
        # Fix 3: Test complementarity with new parameters
        epsilon = 1e-6
        r0, c0 = 2.0, 0.5
        
        d2profit_new = (test_profit(r0 + epsilon, c0 + epsilon, suggested_params) - 
                       test_profit(r0 + epsilon, c0 - epsilon, suggested_params) - 
                       test_profit(r0 - epsilon, c0 + epsilon, suggested_params) + 
                       test_profit(r0 - epsilon, c0 - epsilon, suggested_params)) / (4 * epsilon**2)
        
        print(f"   FIXED COMPLEMENTARITY: ∂²π/∂r∂c = {d2profit_new:.6f}")
        print(f"   (Positive complementarity: {d2profit_new > 0})")
        
        print("\n3. Recommended Parameter Set:")
        for k, v in suggested_params.items():
            print(f"   {k} = {v}")
        
        return suggested_params
    
    def run_full_diagnosis(self):
        """
        Complete diagnostic analysis
        """
        print("MATHEMATICAL MODEL DIAGNOSTIC ANALYSIS")
        print("=" * 70)
        
        # Analyze each component
        snr_results = self.analyze_snr_issues()
        game_results = self.analyze_game_theory_issues()
        fixed_params = self.suggest_parameter_fixes()
        
        print("\n" + "=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)
        
        print("ROOT CAUSES:")
        print("1. SNR optimization: Mathematical formulation may need constraints")
        print("2. Game equilibrium: Cost parameters too low relative to benefits")
        print("3. Complementarity: Need to check cross-partial derivative calculation")
        print("4. Strategic deterrence: No interior solution to vary with detection")
        
        print("\nNEXT STEPS:")
        print("1. Use the suggested parameter set above")
        print("2. Add bounds/constraints to optimization problems")
        print("3. Check that your theoretical model assumptions hold")
        print("4. Consider whether the mathematical formulation matches economic intuition")
        
        return fixed_params

# Run diagnosis
if __name__ == "__main__":
    diagnostic = DiagnosticVerification()
    fixed_params = diagnostic.run_full_diagnosis()
    
    print(f"\n💡 Try rerunning your verification with these parameters:")
    print(f"   Especially increase k (rush order cost) and l (cancellation cost)")
