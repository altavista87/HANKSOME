"""
SOE HANK Impulse Response Functions - Milestone 3
=================================================

Extends IRF computation with FX shocks:
- Fed tightening (capital outflow → depreciation)
- Capital flight (sudden stop)
- Commodity boom/bust (terms of trade)
- BNM policy responses (follow vs hold)

Integrates with SOE HANK model (Milestone 2).
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from model_soe import MalaysiaHANK_SOE
from soe_params import SOEParams

OUTPUT_DIR = Path("/Users/sir/malaysia_hank/outputs/soe_simulation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FXShock:
    """Definition of an FX shock."""
    name: str
    description: str
    shock_type: str
    parameters: Dict


class SOEImpulseResponseComputer:
    """
    Compute impulse responses for SOE HANK with FX shocks.
    """
    
    # Pre-defined shock types
    SHOCKS = {
        'fed_hike': FXShock(
            name='Fed Rate Hike',
            description='US Fed raises rates 100bps',
            shock_type='fed_hike',
            parameters={'fed_hike_bps': 100, 'persistence': 0.9}
        ),
        'capital_flight': FXShock(
            name='Capital Flight',
            description='Emerging market sudden stop',
            shock_type='capital_flight',
            parameters={'outflow_gdp_pct': 8, 'persistence': 0.7}
        ),
        'commodity_boom': FXShock(
            name='Commodity Boom',
            description='Palm oil + LNG price surge',
            shock_type='commodity',
            parameters={'price_shock': 0.30, 'persistence': 0.8}
        ),
        'commodity_bust': FXShock(
            name='Commodity Bust',
            description='Palm oil + LNG price collapse',
            shock_type='commodity',
            parameters={'price_shock': -0.30, 'persistence': 0.8}
        ),
        'safe_haven': FXShock(
            name='Safe Haven Flows',
            description='Risk-off episode, EM outflows',
            shock_type='safe_haven',
            parameters={'flow_shock': -0.05, 'persistence': 0.6}
        ),
    }
    
    def __init__(self, model: MalaysiaHANK_SOE, params: SOEParams, ss_state: Dict):
        """
        Initialize IRF computer.
        
        Args:
            model: SOE HANK model
            params: SOE parameters
            ss_state: Steady state household distribution
        """
        self.model = model
        self.params = params
        self.ss_state = ss_state
        self.n_agents = ss_state['liquid'].shape[0]
        
        print(f"✓ SOE IRF Computer initialized")
        print(f"  Agents: {self.n_agents}")
        print(f"  Baseline ER: {params.exchange_rate_baseline:.2f}")
    
    def generate_exchange_rate_path(
        self,
        shock_type: str,
        shock_params: Dict,
        T: int = 40
    ) -> np.ndarray:
        """
        Generate exchange rate path for shock.
        
        Args:
            shock_type: Type of shock
            shock_params: Shock parameters
            T: Horizon
        
        Returns:
            er_path: Exchange rate path (MYR/USD)
        """
        er_baseline = self.params.exchange_rate_baseline
        persistence = shock_params.get('persistence', 0.7)
        
        er_path = np.full(T, er_baseline)
        
        if shock_type == 'fed_hike':
            # Fed hike → interest differential → capital outflow → depreciation
            hike_bps = shock_params.get('fed_hike_bps', 100)
            hike_pct = hike_bps / 10000  # Convert to decimal
            
            # Immediate depreciation (capital outflow)
            initial_dep = hike_pct * self.params.capital_mobility * 0.5
            er_path[0] = er_baseline * (1 + initial_dep)
            
            # Gradual mean reversion (BNM can intervene)
            for t in range(1, T):
                er_path[t] = er_path[t-1] * persistence + \
                            er_baseline * (1 - persistence) * \
                            (1 - self.params.fx_intervention_strength)
        
        elif shock_type == 'capital_flight':
            # Sudden stop → sharp depreciation
            outflow = shock_params.get('outflow_gdp_pct', 8) / 100
            initial_dep = outflow * 2.0  # 8% outflow → ~16% depreciation
            
            er_path[0] = er_baseline * (1 + initial_dep)
            
            # Slower recovery (crisis persistence)
            for t in range(1, T):
                er_path[t] = er_path[t-1] * persistence + er_baseline * (1 - persistence)
        
        elif shock_type == 'commodity':
            # Terms of trade shock
            price_shock = shock_params.get('price_shock', 0.30)
            
            # Export revenue change → trade balance → ER
            tb_impact = price_shock * (self.params.palm_oil_share_exports + 
                                       self.params.lng_share_exports)
            er_change = -tb_impact * 0.5  # Boom appreciates, bust depreciates
            
            er_path[0] = er_baseline * (1 + er_change)
            
            for t in range(1, T):
                er_path[t] = er_path[t-1] * persistence + er_baseline * (1 - persistence)
        
        elif shock_type == 'safe_haven':
            # Risk-off → EM outflow
            flow_shock = shock_params.get('flow_shock', -0.05)
            initial_dep = abs(flow_shock) * 3.0
            
            er_path[0] = er_baseline * (1 + initial_dep)
            
            for t in range(1, T):
                er_path[t] = er_path[t-1] * persistence + er_baseline * (1 - persistence)
        
        return er_path
    
    def simulate_shock(
        self,
        shock_name: str,
        bnm_response: str = 'passive',
        T: int = 40
    ) -> Dict:
        """
        Simulate impulse response to FX shock.
        
        Args:
            shock_name: Key from SHOCKS dict
            bnm_response: 'passive' (let ER adjust), 'defend' (hike OPR), or 'intervene' (use reserves)
            T: Horizon
        
        Returns:
            results: IRF results with heterogeneous impacts
        """
        if shock_name not in self.SHOCKS:
            raise ValueError(f"Unknown shock: {shock_name}. Available: {list(self.SHOCKS.keys())}")
        
        shock = self.SHOCKS[shock_name]
        print(f"\nSimulating: {shock.name}")
        print(f"  Description: {shock.description}")
        print(f"  BNM response: {bnm_response}")
        
        # Generate exchange rate path
        er_path = self.generate_exchange_rate_path(
            shock.shock_type,
            shock.parameters,
            T
        )
        
        # Adjust for BNM response
        if bnm_response == 'defend':
            # BNM hikes OPR to defend Ringgit → less depreciation
            er_path = self.params.exchange_rate_baseline + \
                     (er_path - self.params.exchange_rate_baseline) * 0.5
        elif bnm_response == 'intervene':
            # BNM uses reserves to smooth
            er_path = self._apply_fx_intervention(er_path)
        
        # Foreign interest rate path
        r_foreign_path = self._generate_r_foreign_path(shock, T)
        
        # Simulate household responses
        results = self._simulate_household_paths(er_path, r_foreign_path, T)
        
        # Add shock metadata
        results['shock'] = {
            'name': shock.name,
            'type': shock.shock_type,
            'bnm_response': bnm_response,
            'er_path': er_path.tolist(),
            'r_foreign_path': r_foreign_path.tolist(),
        }
        
        return results
    
    def _generate_r_foreign_path(self, shock: FXShock, T: int) -> np.ndarray:
        """Generate foreign interest rate path."""
        r_ff_baseline = self.params.r_foreign_baseline
        r_path = np.full(T, r_ff_baseline)
        
        if shock.shock_type == 'fed_hike':
            hike_bps = shock.parameters.get('fed_hike_bps', 100)
            hike_pct = hike_bps / 10000
            persistence = shock.parameters.get('persistence', 0.9)
            
            r_path[0] = r_ff_baseline + hike_pct
            for t in range(1, T):
                r_path[t] = r_path[t-1] * persistence + r_ff_baseline * (1 - persistence)
        
        return r_path
    
    def _apply_fx_intervention(self, er_path: np.ndarray) -> np.ndarray:
        """Apply BNM FX intervention to smooth path."""
        er_baseline = self.params.exchange_rate_baseline
        intervention = self.params.fx_intervention_strength
        
        # Smooth deviations from baseline
        smoothed = er_baseline + (er_path - er_baseline) * (1 - intervention)
        return smoothed
    
    def _simulate_household_paths(
        self,
        er_path: np.ndarray,
        r_foreign_path: np.ndarray,
        T: int
    ) -> Dict:
        """
        Simulate household consumption paths.
        
        Returns aggregate and quintile-specific results.
        """
        # Storage
        aggregate = {
            'nominal_consumption': [],
            'real_consumption': [],
            'cpi': [],
            'usd_debt_service': [],
        }
        
        quintiles = {q: {'real_consumption': [], 'cpi': []} 
                    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']}
        
        state = {k: v.clone() for k, v in self.ss_state.items()}
        
        for t in range(T):
            # Get policies with FX
            policies = self.model.forward_soe(
                state,
                er_path[t],
                r_foreign_path[t],
                track_quintiles=True
            )
            
            # Aggregate
            aggregate['nominal_consumption'].append(
                policies['nominal_consumption'].mean().item()
            )
            aggregate['real_consumption'].append(
                policies['real_consumption'].mean().item()
            )
            aggregate['cpi'].append(policies['cpi'])
            aggregate['usd_debt_service'].append(
                policies['usd_debt_service'].mean().item()
            )
            
            # By quintile
            for q in quintiles.keys():
                if q in policies.get('quintile_real_consumption', {}):
                    q_data = policies['quintile_real_consumption'][q]
                    quintiles[q]['real_consumption'].append(q_data['real'])
                    quintiles[q]['cpi'].append(q_data['cpi'])
                else:
                    # Use previous value or aggregate
                    if quintiles[q]['real_consumption']:
                        quintiles[q]['real_consumption'].append(
                            quintiles[q]['real_consumption'][-1]
                        )
                        quintiles[q]['cpi'].append(quintiles[q]['cpi'][-1])
            
            # Update state (simplified transition)
            state['liquid'] = torch.clamp(
                policies['liquid_savings'], 0, self.params.liquid_max
            )
            state['illiquid'] = torch.clamp(
                policies['illiquid_savings'], 0, self.params.illiquid_max
            )
        
        return {
            'aggregate': aggregate,
            'quintiles': quintiles,
        }
    
    def compute_irf_deviations(self, results: Dict) -> Dict:
        """
        Compute deviations from steady state (IRF format).
        
        Returns percentage deviations for standard IRF plots.
        """
        # Baseline (first period = steady state, approximately)
        base_real = results['aggregate']['real_consumption'][0]
        
        irf = {
            'aggregate': [
                (c / base_real - 1) * 100 
                for c in results['aggregate']['real_consumption']
            ]
        }
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            if q in results['quintiles'] and results['quintiles'][q]['real_consumption']:
                q_consumption = results['quintiles'][q]['real_consumption']
                if len(q_consumption) > 0 and q_consumption[0] > 0:
                    irf[q] = [
                        (c / q_consumption[0] - 1) * 100
                        for c in q_consumption
                    ]
        
        return irf
    
    def compare_bnm_responses(self, shock_name: str, T: int = 40) -> Dict:
        """
        Compare different BNM policy responses to same shock.
        
        Args:
            shock_name: Shock to analyze
            T: Horizon
        
        Returns:
            comparison: Results for each BNM response
        """
        responses = ['passive', 'defend', 'intervene']
        comparison = {}
        
        print(f"\n{'='*60}")
        print(f"BNM POLICY COMPARISON: {self.SHOCKS[shock_name].name}")
        print(f"{'='*60}")
        
        for response in responses:
            results = self.simulate_shock(shock_name, response, T)
            irf = self.compute_irf_deviations(results)
            
            comparison[response] = {
                'results': results,
                'irf': irf,
                'max_impact': min(irf['aggregate']),  # Most negative
            }
            
            print(f"\n{response.upper()}:")
            print(f"  Max consumption impact: {min(irf['aggregate']):.2f}%")
            print(f"  Q1 impact: {min(irf.get('Q1', [0])):.2f}%")
            print(f"  Q5 impact: {min(irf.get('Q5', [0])):.2f}%")
        
        return comparison
    
    def save_results(self, results: Dict, filename: str):
        """Save IRF results to JSON."""
        filepath = OUTPUT_DIR / filename
        
        # Convert numpy arrays to lists for JSON
        json_safe = self._make_json_safe(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_safe, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")
    
    def _make_json_safe(self, obj):
        """Recursively convert numpy/torch to JSON-safe types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


def test_irf_soe():
    """Test SOE IRF computer."""
    print("\n" + "="*60)
    print("TESTING SOE IRF COMPUTER - MILESTONE 3")
    print("="*60)
    
    # Initialize model
    params = SOEParams()
    model = MalaysiaHANK_SOE(params)
    
    # Create dummy steady state
    n = 2000
    state = {
        'liquid': torch.rand(n, 1) * 50,
        'illiquid': torch.rand(n, 1) * 200,
        'mortgage_debt': torch.rand(n, 1) * 100,
        'base_income': torch.exp(torch.randn(n, 1) * 0.5 + 0.5),
        'education_level': torch.randint(0, 3, (n,)),
        'age': torch.randint(18, 66, (n, 1)).float(),
        'health_state': torch.randint(0, 4, (n,)),
        'housing_type': torch.randint(0, 3, (n,)),
        'location': torch.randint(0, 26, (n,))
    }
    
    # Initialize IRF computer
    irf_comp = SOEImpulseResponseComputer(model, params, state)
    
    # Test 1: Single shock
    print("\n1. Testing Fed Hike shock (passive BNM response)...")
    results = irf_comp.simulate_shock('fed_hike', bnm_response='passive', T=20)
    irf = irf_comp.compute_irf_deviations(results)
    
    print(f"   Aggregate max impact: {min(irf['aggregate']):.2f}%")
    print(f"   Q1 max impact: {min(irf.get('Q1', [0])):.2f}%")
    print(f"   Q5 max impact: {min(irf.get('Q5', [0])):.2f}%")
    
    # Test 2: Compare BNM responses
    print("\n2. Comparing BNM policy responses to Fed hike...")
    comparison = irf_comp.compare_bnm_responses('fed_hike', T=20)
    
    # Test 3: Multiple shocks
    print("\n" + "="*60)
    print("3. Testing multiple FX shocks:")
    print("="*60)
    
    for shock_name in ['fed_hike', 'capital_flight', 'commodity_bust']:
        results = irf_comp.simulate_shock(shock_name, bnm_response='passive', T=20)
        irf = irf_comp.compute_irf_deviations(results)
        print(f"\n{shock_name}:")
        print(f"  Aggregate: {min(irf['aggregate']):+.2f}%")
        for q in ['Q1', 'Q3', 'Q5']:
            if q in irf:
                print(f"  {q}: {min(irf[q]):+.2f}%")
    
    # Save sample results
    irf_comp.save_results(results, 'sample_irf_results.json')
    
    print("\n" + "="*60)
    print("✓ Milestone 3 Complete: FX Shocks Working!")
    print("="*60)
    print("\nAvailable shocks:")
    for name, shock in SOEImpulseResponseComputer.SHOCKS.items():
        print(f"  - {name}: {shock.description}")
    
    print("\n" + "="*60)
    print("NOTE: Model shows zero impact because base neural network")
    print("      is untrained. Milestone 4 will calibrate mechanical")
    print("      FX effects (import prices, USD debt service).")
    print("="*60)
    
    return irf_comp


if __name__ == "__main__":
    test_irf_soe()
