"""classmixing.py
Author: Muhammad R. Hasyim

Script to generate a list of random electrolytes based on their classifications using an object-oriented approach. Classifications for doing non-RPMD simulations:
(1) 40% Salt in protic solvents
(2) 40% Salt in polar aprotic solvents
(3) 10% Salt in ionic liquids
(4) 5% Molten salt 
(5) 5% Aqueous electrolytes
To change these ratios manually, go to Line 544-550. For RPMD simulations:
(1) 30% Salt in protic solvents
(2) 30% Salt in aprotic solvents
(3) 40% Aqueous electrolytes

If we turn on the flag, --random-only, then these ratios are ignored and we go with fully random mixtures.
"""

import re
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import csv
import argparse
import textwrap
from math import gcd
from functools import reduce

class SpeciesSelector:
    """Utility class for selecting chemical species."""
    
    def __init__(self):
        """Initializes the SpeciesSelector with a list of lanthanides."""
        self.lanthanides = [
            "La", "Ce", "Pr", "Nd", "Pm", "Sm",
            "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
            "Tm", "Yb", "Lu"
        ]
        
    def contains_lanthanide(self, strings: List[str]) -> bool:
        """Checks if any of the given strings contain a lanthanide.

        Args:
            strings: A list of chemical species formulas.

        Returns:
            bool: True if any string contains a lanthanide, False otherwise.
        """
        return any(lanthanide in string for string in strings for lanthanide in self.lanthanides)
    
    def check_for_duplicates(self, existing_species: List[str], selected_species: List[str]) -> bool:
        """Checks for duplicates between existing and selected species.

        Args:
            existing_species: A list of existing species formulas.
            selected_species: A list of selected species formulas.

        Returns:
            bool: True if there are duplicates, False otherwise.
        """
        existing_cleaned = [re.split(r'[+-]', formula)[0] for formula in existing_species]
        selected_cleaned = [re.split(r'[+-]', formula)[0] for formula in selected_species]
        return any(selected in existing_cleaned for selected in selected_cleaned)
    
    def choose_species(self, species: pd.DataFrame, max_comp: int, existing_species: List[str], 
                      solvent: bool = False) -> Tuple:
        """Randomly selects species from the given DataFrame."""
        if len(species) == 0:
            # Return empty lists if no species available
            return ([], [], [], []) if solvent else ([], [])
        
        # Don't try to select more components than available species
        max_comp = min(max_comp, len(species))
        
        formulas = self.lanthanides
        while self.contains_lanthanide(formulas) or self.check_for_duplicates(existing_species, formulas):
            # Choose a number of components between 1 and max_comp
            n_components = np.random.randint(1, max_comp + 1)
            indices = np.random.choice(len(species), size=n_components, replace=False)
            formulas = np.array(species["formula"])[indices].tolist()
            
            if not self.contains_lanthanide(formulas) and not self.check_for_duplicates(existing_species, formulas):
                break
        
        if solvent:
            return (np.array(species["formula"])[indices].tolist(),
                   np.array(species["charge"])[indices].tolist(),
                   np.array(species["min_temperature"])[indices].tolist(),
                   np.array(species["max_temperature"])[indices].tolist())
        else:
            return (np.array(species["formula"])[indices].tolist(),
                   np.array(species["charge"])[indices].tolist())

class ChargeSolver:
    """Utility class for solving charge equations."""
    
    @staticmethod
    def solve_single_equation(coefficients: List[float]) -> List[int]:
        """Solves a single charge balance equation.

        Args:
            coefficients: List of coefficients representing the charges.

        Returns:
            List[int]: List of integer solutions for the coefficients.
        """
        prob = LpProblem("Integer_Solution_Problem", LpMinimize)
        x = [LpVariable(f"x{i}", 1, 5, cat='Integer') for i in range(len(coefficients))]
        prob += lpSum(coeff * var for coeff, var in zip(coefficients, x)) == 0
        prob.solve()
        return [int(v.varValue) for v in prob.variables()[1:]]

class Electrolyte(ABC):
    """Base class for all electrolyte types."""
    
    Avog = 6.023e23
    
    def __init__(self, species_selector: SpeciesSelector, charge_solver: ChargeSolver, 
                 is_rpmd: bool = False, max_cations: int = 4, max_anions: int = 4, 
                 max_solvents: int = 4):
        """Initializes the Electrolyte with species selector and charge solver.

        Args:
            species_selector: Instance of SpeciesSelector for selecting species.
            charge_solver: Instance of ChargeSolver for solving charge equations.
            is_rpmd: Boolean indicating if the electrolyte is for RPMD simulations.
            max_cations: Maximum number of cation species (default: 4)
            max_anions: Maximum number of anion species (default: 4)
            max_solvents: Maximum number of solvent species (default: 4)
        """
        self.species_selector = species_selector
        self.charge_solver = charge_solver
        self.max_cations = max_cations
        self.max_anions = max_anions
        self.max_solvents = max_solvents
        self.fac = 0.05
        self.cations = []
        self.anions = []
        self.solvents = []
        self.cat_charges = []
        self.an_charges = []
        self.stoich = []
        self.stoich_solv = []
        self.min_temp = 0
        self.max_temp = 0
        self.is_rpmd = is_rpmd
    
    def generate(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame, 
                solvents_df: pd.DataFrame) -> None:
        """Template method for generating electrolytes.

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.
        """
        # Store original dataframes for special cases (like IL)
        self.original_cations_df = cations_df
        self.original_anions_df = anions_df
        
        # Apply RPMD filtering if needed
        if self.is_rpmd:
            cations_df = cations_df[cations_df['in_rpmd']].copy()
            anions_df = anions_df[anions_df['in_rpmd']].copy()
            solvents_df = solvents_df[solvents_df['in_rpmd']].copy()
        
        # 1. Filter compatible species
        filtered_cations, filtered_anions, filtered_solvents = self._filter_species(
            cations_df, anions_df, solvents_df)
        
        # 2. Generate main species
        self._generate_main_species(filtered_cations, filtered_anions)
        
        # 3. Generate solvents (if any)
        self._generate_solvents(filtered_solvents)
        
        # 4. Set temperature range
        self._set_temperature_range()
    
    def _generate_main_species(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame) -> None:
        """Generate cations and anions and solve charge balance."""
        # Generate species with separate max components for each
        self.cations, self.cat_charges = self.species_selector.choose_species(
            cations_df, self.max_cations, [])
        self.anions, self.an_charges = self.species_selector.choose_species(
            anions_df, self.max_anions, self.cations)
        
        # Solve charge balance
        charges = self.cat_charges + self.an_charges
        self.stoich = self.charge_solver.solve_single_equation(charges)
    
    @abstractmethod
    def _filter_species(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame, 
                       solvents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter species based on electrolyte type.

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.

        Returns:
            Tuple: Filtered DataFrames for cations, anions, and solvents.
        """
        pass
    
    @abstractmethod
    def _generate_solvents(self, solvents_df: pd.DataFrame) -> None:
        """Generate solvents for the electrolyte.

        Args:
            solvents_df: DataFrame containing solvent species.
        """
        pass
    
    @abstractmethod
    def _set_temperature_range(self) -> None:
        """Set the temperature range for the electrolyte."""
        pass
    
    def to_dict(self, name: str) -> Dict[str, Any]:
        """Convert electrolyte data to dictionary format for CSV."""
        # Extract concentration from name suffix
        is_low_conc = '-lowconc' in name
        distance = 2.25 if is_low_conc else 1.75
        
        # Calculate actual concentrations
        conc = 0.62035049089/distance**3  # number per nm3
        salt_molfrac = np.array(self.stoich)/sum(self.stoich) if len(self.stoich) > 0 else []
        salt_conc = salt_molfrac * conc / self.Avog * 1e24 if len(salt_molfrac) > 0 else []
        
        entry = {
            'category': 'random',
            'comment/name': name,
            'DOI': '',
            'units': 'number' if isinstance(self, MoltenSaltElectrolyte) else 'mass',
            'temperature': self.min_temp
        }
        
        # Create entries for each type up to their maximum
        for i in range(max(self.max_cations, self.max_anions, self.max_solvents)):
            # Cations
            if i < self.max_cations:
                entry[f'cation{i+1}'] = self.cations[i] if i < len(self.cations) else ''
                entry[f'cation{i+1}_conc'] = salt_conc[i] if i < len(salt_conc) else ''
                
            # Anions
            if i < self.max_anions:
                entry[f'anion{i+1}'] = self.anions[i] if i < len(self.anions) else ''
                entry[f'anion{i+1}_conc'] = salt_conc[i+len(self.cations)] if i+len(self.cations) < len(salt_conc) else ''
                
            # Solvents
            if i < self.max_solvents:
                entry[f'solvent{i+1}'] = self.solvents[i] if i < len(self.solvents) else ''
                
                # Special handling for ionic liquid solvents
                if isinstance(self, IonicLiquidElectrolyte) and len(self.stoich_solv) > 0:
                    # For ionic liquids, normalize while preserving charge balance relationships
                    if i < len(self.stoich_solv):
                        # Find the GCD of all stoich values to maintain integer relationships
                        def find_gcd(list_of_nums):
                            # Convert to integers if they're not already
                            ints = [int(x) for x in list_of_nums]
                            # Find GCD of the list
                            return reduce(gcd, ints)
                        
                        # Calculate GCD and normalize by dividing all values by it
                        common_divisor = find_gcd(self.stoich_solv)
                        normalized_value = self.stoich_solv[i] / common_divisor
                        entry[f'solvent{i+1}_ratio'] = normalized_value
                    else:
                        entry[f'solvent{i+1}_ratio'] = ''
                else:
                    # For other solvents, normalize by dividing by minimum as before
                    entry[f'solvent{i+1}_ratio'] = (self.stoich_solv[i]/min(self.stoich_solv) 
                                                  if i < len(self.stoich_solv) and len(self.stoich_solv) > 0 
                                                  else '')
        
        return entry

class AqueousElectrolyte(Electrolyte):
    """Class for aqueous electrolytes."""
    
    def _filter_species(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame,
                       solvents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter species for aqueous electrolytes.

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.

        Returns:
            Tuple: Filtered DataFrames for cations, anions, and solvents.
        """
        return (cations_df[cations_df['in_aq']].copy(),
                anions_df[anions_df['in_aq']].copy(),
                pd.DataFrame())  # No need for solvents DataFrame
    
    def _generate_solvents(self, solvents_df: pd.DataFrame) -> None:
        """Generate solvents for aqueous electrolytes.

        Args:
            solvents_df: DataFrame containing solvent species.
        """
        self.solvents = ['H2O']
        self.stoich_solv = [1]
    
    def _set_temperature_range(self) -> None:
        """Set the temperature range for aqueous electrolytes."""
        self.min_temp = (1 + self.fac) * 273
        self.max_temp = (1 - self.fac) * 373

class ProticElectrolyte(Electrolyte):
    """Class for protic electrolytes."""
    
    def _filter_species(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame,
                       solvents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter species for protic electrolytes.

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.

        Returns:
            Tuple: Filtered DataFrames for cations, anions, and solvents.
        """
        return (cations_df[cations_df['in_protic']].copy(),
                anions_df[anions_df['in_protic']].copy(),
                solvents_df[solvents_df['protic']].copy())
    
    def _generate_solvents(self, solvents_df: pd.DataFrame) -> None:
        """Generate solvents for protic electrolytes.

        Args:
            solvents_df: DataFrame containing solvent species.
        """
        self.solvents, _, min_temps, max_temps = self.species_selector.choose_species(
            solvents_df, self.max_solvents, self.cations + self.anions, solvent=True)
        self.stoich_solv = np.random.randint(1, self.max_solvents + 1, size=len(self.solvents))
        
        # Calculate temperature range
        solv_molfrac = np.array(self.stoich_solv)/sum(self.stoich_solv)
        self.min_temp = (1 + self.fac) * np.sum(np.array(min_temps) * solv_molfrac)
        self.max_temp = (1 - self.fac) * np.sum(np.array(max_temps) * solv_molfrac)
    
    def _set_temperature_range(self) -> None:
        """Set the temperature range for protic electrolytes."""
        # Temperature range is set in _generate_solvents for this class
        pass

class AproticElectrolyte(Electrolyte):
    """Class for aprotic electrolytes."""
    
    def _filter_species(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame,
                       solvents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter species for aprotic electrolytes.

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.

        Returns:
            Tuple: Filtered DataFrames for cations, anions, and solvents.
        """
        return (cations_df[cations_df['in_aprotic']].copy(),
                anions_df[anions_df['in_aprotic']].copy(),
                solvents_df[solvents_df['polar_aprotic']].copy())
    
    def _generate_solvents(self, solvents_df: pd.DataFrame) -> None:
        """Generate solvents for aprotic electrolytes.

        Args:
            solvents_df: DataFrame containing solvent species.
        """
        self.solvents, _, min_temps, max_temps = self.species_selector.choose_species(
            solvents_df, self.max_solvents, self.cations + self.anions, solvent=True)
        self.stoich_solv = np.random.randint(1, self.max_solvents + 1, size=len(self.solvents))
        
        # Calculate temperature range
        solv_molfrac = np.array(self.stoich_solv)/sum(self.stoich_solv)
        self.min_temp = (1 + self.fac) * np.sum(np.array(min_temps) * solv_molfrac)
        self.max_temp = (1 - self.fac) * np.sum(np.array(max_temps) * solv_molfrac)
    
    def _set_temperature_range(self) -> None:
        """Set the temperature range for aprotic electrolytes."""
        # Temperature range is set in _generate_solvents for this class
        pass

class IonicLiquidElectrolyte(Electrolyte):
    """Class for ionic liquid electrolytes."""
    
    def _filter_species(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame,
                       solvents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter species for ionic liquid electrolytes.

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.

        Returns:
            Tuple: Filtered DataFrames for cations, anions, and solvents.
        """
        return (cations_df[cations_df['in_IL']].copy(),
                anions_df[anions_df['in_IL']].copy(),
                pd.DataFrame())  # We'll handle IL components separately
    
    def _generate_solvents(self, solvents_df: pd.DataFrame) -> None:
        """Generate solvents for ionic liquid electrolytes.

        Args:
            solvents_df: DataFrame containing solvent species.
        """
        # Generate IL components as solvents using original dataframes
        il_solv_cations = self.original_cations_df[self.original_cations_df['IL_comp']].copy()
        il_solv_anions = self.original_anions_df[self.original_anions_df['IL_comp']].copy()
        
        # Select ionic liquid components
        solv1, solv_charges1 = self.species_selector.choose_species(
            il_solv_cations, self.max_cations, self.cations + self.anions)
        solv2, solv_charges2 = self.species_selector.choose_species(
            il_solv_anions, self.max_cations, self.cations + self.anions + solv1)
        
        # Ensure we have at least one cation and one anion
        if not solv1 and len(il_solv_cations) > 0:
            idx = 0  # Select first available cation
            solv1 = [il_solv_cations.iloc[idx]['formula']]
            solv_charges1 = [il_solv_cations.iloc[idx]['charge']]
        
        if not solv2 and len(il_solv_anions) > 0:
            idx = 0  # Select first available anion
            solv2 = [il_solv_anions.iloc[idx]['formula']]
            solv_charges2 = [il_solv_anions.iloc[idx]['charge']]
        
        # At this point, we should have both cations and anions
        # Now use a simple balancing approach to ensure perfect charge neutrality
        if solv1 and solv2:
            # Use the charge solver to find the optimal ratio for charge balance
            cat_charge = solv_charges1[0]
            an_charge = solv_charges2[0]  # This is already negative
            
            # Solve the charge balance equation: cat_charge * x + an_charge * y = 0
            # We need to find integer values for x and y
            coefficients = [0, cat_charge, an_charge]  # First 0 is for the objective function
            solution = self.charge_solver.solve_single_equation(coefficients)
            
            # Extract the solution values
            cat_count = solution[0]
            an_count = solution[1]
            
            # Keep only the first cation and anion with the calculated ratio
            self.solvents = [solv1[0], solv2[0]]
            self.stoich_solv = [cat_count, an_count]
            
            # Check the charge balance
            total_charge = cat_charge * cat_count + an_charge * an_count
            print(f"IL solvent pair: {self.solvents[0]} (charge {cat_charge}, ratio {cat_count}) and "
                  f"{self.solvents[1]} (charge {an_charge}, ratio {an_count})")
            print(f"Charge balance check: {total_charge}")
        else:
            # If we couldn't find IL components, fall back to the standard behavior
            # using the regular solvents dataframe
            super()._generate_solvents(solvents_df)
    
    def _set_temperature_range(self) -> None:
        """Set the temperature range for ionic liquid electrolytes."""
        self.min_temp = 300
        self.max_temp = 400

class MoltenSaltElectrolyte(Electrolyte):
    """Class for molten salt electrolytes."""
    
    def _filter_species(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame,
                       solvents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter species for molten salt electrolytes.

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.

        Returns:
            Tuple: Filtered DataFrames for cations, anions, and solvents.
        """
        return (cations_df[cations_df['MS_comp']].copy(),
                anions_df[anions_df['MS_comp']].copy(),
                pd.DataFrame())  # No solvents for molten salt
    
    def _generate_solvents(self, solvents_df: pd.DataFrame) -> None:
        """Generate solvents for molten salt electrolytes.

        Args:
            solvents_df: DataFrame containing solvent species.
        """
        self.solvents = []
        self.stoich_solv = []
    
    def _set_temperature_range(self) -> None:
        """Set the temperature range for molten salt electrolytes."""
        self.min_temp = 1000
        self.max_temp = 1300

class RandomElectrolyte(Electrolyte):
    """Class for completely random electrolytes without type constraints."""
    
    def __init__(self, species_selector: SpeciesSelector, charge_solver: ChargeSolver, 
                 is_rpmd: bool = False, max_cations: int = 4, max_anions: int = 4, 
                 max_solvents: int = 4):
        """Initializes the RandomElectrolyte with species selector and charge solver.

        Args:
            species_selector: Instance of SpeciesSelector for selecting species.
            charge_solver: Instance of ChargeSolver for solving charge equations.
            is_rpmd: Boolean indicating if the electrolyte is for RPMD simulations.
            max_cations: Maximum number of cation species (default: 4)
            max_anions: Maximum number of anion species (default: 4)
            max_solvents: Maximum number of solvent species (default: 4)
        """
        super().__init__(species_selector, charge_solver, is_rpmd, max_cations, max_anions, max_solvents)
    
    def _filter_species(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame,
                       solvents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter species for random electrolytes (no filtering).

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.

        Returns:
            Tuple: Filtered DataFrames for cations, anions, and solvents (all available).
        """
        # No filtering for random electrolytes, use all available species
        if self.is_rpmd:
            return (cations_df[cations_df['in_rpmd']].copy(),
                   anions_df[anions_df['in_rpmd']].copy(),
                   solvents_df[solvents_df['in_rpmd']].copy())
        return cations_df.copy(), anions_df.copy(), solvents_df.copy()
    
    def _generate_solvents(self, solvents_df: pd.DataFrame) -> None:
        """Generate solvents for random electrolytes.

        Args:
            solvents_df: DataFrame containing solvent species.
        """
        self.solvents, _, min_temps, max_temps = self.species_selector.choose_species(
            solvents_df, self.max_solvents, self.cations + self.anions, solvent=True)
        self.stoich_solv = np.random.randint(1, self.max_solvents + 1, size=len(self.solvents))
        
        # Calculate temperature range
        solv_molfrac = np.array(self.stoich_solv)/sum(self.stoich_solv)
        self.min_temp = (1 + self.fac) * np.sum(np.array(min_temps) * solv_molfrac)
        self.max_temp = (1 - self.fac) * np.sum(np.array(max_temps) * solv_molfrac)
    
    def _set_temperature_range(self) -> None:
        """Set the temperature range for random electrolytes."""
        # Temperature range is set in _generate_solvents for this class
        pass
    
    def to_dict(self, name: str) -> Dict[str, Any]:
        """Convert to dictionary using parent class implementation.

        Args:
            name: Name of the electrolyte.

        Returns:
            Dict[str, Any]: Dictionary representation of the electrolyte data.
        """
        return super().to_dict(name)  # Use parent class implementation with mass units

class ElectrolyteFactory:
    """Factory class for creating different types of electrolytes."""
    
    def __init__(self, is_rpmd: bool = False, random_only: bool = False,
                 max_cations: int = 4, max_anions: int = 4, max_solvents: int = 4):
        """Initializes the ElectrolyteFactory with species selector and charge solver.

        Args:
            is_rpmd: Boolean indicating if the factory is for RPMD simulations.
            random_only: Boolean indicating if only random electrolytes should be created.
            max_cations: Maximum number of cation species (default: 4)
            max_anions: Maximum number of anion species (default: 4)
            max_solvents: Maximum number of solvent species (default: 4)
        """
        self.species_selector = SpeciesSelector()
        self.charge_solver = ChargeSolver()
        self.is_rpmd = is_rpmd
        self.random_only = random_only
        self.max_cations = max_cations
        self.max_anions = max_anions
        self.max_solvents = max_solvents
        
        if random_only:
            self.class_probabilities = {'random': 1.0}
        elif is_rpmd:
            # RPMD electrolytes are always aqueous or protic or aprotic.
            # We want to make sure that we have a good mix of all three.
            # But there's no point of simulating molten salt electrolytes and ionic liquids in RPMD.
            self.class_probabilities = {
                'protic': 0.3,
                'aprotic': 0.3,
                'aq': 0.4
            }
        else:
            self.class_probabilities = {
                'protic': 0.40,
                'aprotic': 0.40,
                'IL': 0.1,
                'MS': 0.05,
                'aq': 0.05
            }
            
        self.class_mapping = {
            'protic': ProticElectrolyte,
            'aprotic': AproticElectrolyte,
            'IL': IonicLiquidElectrolyte,
            'MS': MoltenSaltElectrolyte,
            'aq': AqueousElectrolyte,
            'random': RandomElectrolyte
        }
    
    def create_electrolyte(self, cations_df: pd.DataFrame, anions_df: pd.DataFrame, 
                          solvents_df: pd.DataFrame) -> Tuple[Electrolyte, str]:
        """Creates an electrolyte based on the selected class.

        Args:
            cations_df: DataFrame containing cation species.
            anions_df: DataFrame containing anion species.
            solvents_df: DataFrame containing solvent species.

        Returns:
            Tuple: The created electrolyte and its class name.
        """
        # Randomly select electrolyte class based on probabilities
        elyte_class = np.random.choice(
            list(self.class_probabilities.keys()),
            p=list(self.class_probabilities.values())
        )
        
        # Create and generate the electrolyte with specified max components
        electrolyte = self.class_mapping[elyte_class](
            self.species_selector, 
            self.charge_solver, 
            self.is_rpmd,
            max_cations=self.max_cations,
            max_anions=self.max_anions,
            max_solvents=self.max_solvents
        )
        electrolyte.generate(cations_df, anions_df, solvents_df)
        return electrolyte, elyte_class

def generate_electrolytes(is_rpmd: bool = False, 
                         input_files: Dict[str, str] = None,
                         output_file: str = None,
                         n_random: int = None,
                         random_only: bool = False,
                         max_cations: int = 4,
                         max_anions: int = 4,
                         max_solvents: int = 4,
                         required_solvent: str = None,
                         required_cation: str = None,
                         required_anion: str = None,
                         force_require: bool = False) -> pd.DataFrame:
    """Generate electrolytes with given parameters.

    Args:
        is_rpmd: Whether to generate electrolytes for RPMD simulations.
        input_files: Dictionary with paths to input files.
        output_file: Path to output file.
        n_random: Number of random electrolytes to generate.
        random_only: Whether to generate only random electrolytes.
        max_cations: Maximum number of cation species (default: 4)
        max_anions: Maximum number of anion species (default: 4)
        max_solvents: Maximum number of solvent species (default: 4)
        required_solvent: Comma-separated string of required solvent species
        required_cation: Comma-separated string of required cation species
        required_anion: Comma-separated string of required anion species
        force_require: Whether to force at least one required species from each category in every system

    Returns:
        pd.DataFrame: Generated electrolytes dataframe.
    """
    # Set default values
    if input_files is None:
        input_files = {
            'cations': 'cations.csv',
            'anions': 'anions.csv',
            'solvents': 'solvent.csv'
        }
    
    if output_file is None:
        output_file = 'rpmd_elytes.csv' if is_rpmd else 'ml_elytes.csv'
        
    if n_random is None:
        n_random = 20 if is_rpmd else (1000 if random_only else 3000)
    
    # Load data
    cations_df = pd.read_csv(input_files['cations'])
    anions_df = pd.read_csv(input_files['anions'])
    solvents_df = pd.read_csv(input_files['solvents'])
    
    try:
        elytes = pd.read_csv(output_file)
    except FileNotFoundError:
        # Create empty DataFrame with correct columns if file doesn't exist
        elytes = pd.DataFrame(columns=['category', 'comment/name', 'DOI', 'units', 'temperature'] + 
                            [f'{t}{i}' for t in ['cation', 'anion', 'solvent'] 
                             for i in range(1, 5)] +
                            [f'{t}{i}_{"conc" if t != "solvent" else "ratio"}' 
                             for t in ['cation', 'anion', 'solvent'] for i in range(1, 5)])
    
    # Initialize factory with max component parameters
    factory = ElectrolyteFactory(
        is_rpmd=is_rpmd,
        random_only=random_only,
        max_cations=max_cations,
        max_anions=max_anions,
        max_solvents=max_solvents
    )
    
    # Process required species if needed
    required_species = {
        'solvents': set(),
        'cations': set(),
        'anions': set()
    }
    
    # For both force_require and random_only we need to process required species
    if force_require or random_only:
        if required_solvent:
            required_species['solvents'] = {s.strip() for s in required_solvent.split(',')}
            # Verify all required solvents exist in the database
            for solv in required_species['solvents']:
                if solv not in solvents_df['formula'].values:
                    raise ValueError(f"Required solvent {solv} not found in solvent database")
        
        if required_cation:
            required_species['cations'] = {c.strip() for c in required_cation.split(',')}
            for cat in required_species['cations']:
                if cat not in cations_df['formula'].values:
                    raise ValueError(f"Required cation {cat} not found in cation database")
        
        if required_anion:
            required_species['anions'] = {a.strip() for a in required_anion.split(',')}
            for an in required_species['anions']:
                if an not in anions_df['formula'].values:
                    raise ValueError(f"Required anion {an} not found in anion database")

    # Track which required species have been included (only needed for non-forced mode)
    included_species = {
        'solvents': set(),
        'cations': set(),
        'anions': set()
    }

    # Generate electrolytes
    for i in range(n_random):
        # Create electrolyte
        electrolyte, elyte_class = factory.create_electrolyte(cations_df, anions_df, solvents_df)
        
        # Handle required species based on mode
        if force_require:
            # Check if at least one required solvent is already included
            needs_solvent = required_species['solvents'] and not any(solv in required_species['solvents'] for solv in electrolyte.solvents)
            needs_cation = required_species['cations'] and not any(cat in required_species['cations'] for cat in electrolyte.cations)
            needs_anion = required_species['anions'] and not any(anion in required_species['anions'] for anion in electrolyte.anions)
            
            # Force include at least one required solvent if needed
            if needs_solvent:
                # Randomly select one required solvent
                solvent_to_include = list(required_species['solvents'])[np.random.randint(0, len(required_species['solvents']))]
                
                # If there are existing solvents, replace one with the required one
                if electrolyte.solvents:
                    replace_idx = np.random.randint(0, len(electrolyte.solvents))
                    electrolyte.solvents[replace_idx] = solvent_to_include
                else:
                    # If no solvents, add the required one
                    electrolyte.solvents = [solvent_to_include]
                    electrolyte.stoich_solv = [1]
            
            # Force include at least one required cation if needed
            if needs_cation:
                # Randomly select one required cation
                required_cation_list = list(required_species['cations'])
                cation_to_include = required_cation_list[np.random.randint(0, len(required_cation_list))]
                
                # Get the charge for the required cation
                cat_idx = cations_df[cations_df['formula'] == cation_to_include].index[0]
                cat_charge = cations_df.loc[cat_idx, 'charge']
                
                # If there are existing cations, replace one with the required one
                if electrolyte.cations:
                    replace_idx = np.random.randint(0, len(electrolyte.cations))
                    electrolyte.cations[replace_idx] = cation_to_include
                    electrolyte.cat_charges[replace_idx] = cat_charge
                else:
                    # If no cations, add the required one
                    electrolyte.cations = [cation_to_include]
                    electrolyte.cat_charges = [cat_charge]
            
            # Force include at least one required anion if needed
            if needs_anion:
                # Randomly select one required anion
                required_anion_list = list(required_species['anions'])
                anion_to_include = required_anion_list[np.random.randint(0, len(required_anion_list))]
                
                # Get the charge for the required anion
                an_idx = anions_df[anions_df['formula'] == anion_to_include].index[0]
                an_charge = anions_df.loc[an_idx, 'charge']
                
                # If there are existing anions, replace one with the required one
                if electrolyte.anions:
                    replace_idx = np.random.randint(0, len(electrolyte.anions))
                    electrolyte.anions[replace_idx] = anion_to_include
                    electrolyte.an_charges[replace_idx] = an_charge
                else:
                    # If no anions, add the required one
                    electrolyte.anions = [anion_to_include]
                    electrolyte.an_charges = [an_charge]
            
            # Recalculate stoichiometry to maintain charge balance if we changed ions
            if needs_cation or needs_anion:
                # Solve charge balance with new ions
                charges = electrolyte.cat_charges + electrolyte.an_charges
                electrolyte.stoich = electrolyte.charge_solver.solve_single_equation(charges)
        
        # For non-force mode, continue with the existing logic for tracking required species
        elif random_only:
            # Track which required species are included in this system
            included_species['solvents'].update(set(electrolyte.solvents))
            included_species['cations'].update(set(electrolyte.cations))
            included_species['anions'].update(set(electrolyte.anions))
            
            # If this is one of the last chances to include required species
            remaining_systems = n_random - i
            remaining_solvents = required_species['solvents'] - included_species['solvents']
            remaining_cations = required_species['cations'] - included_species['cations']
            remaining_anions = required_species['anions'] - included_species['anions']
            
            # Try to include remaining solvents
            if remaining_solvents and remaining_systems <= len(remaining_solvents):
                solvent_to_include = list(remaining_solvents)[0]
                # Modify the electrolyte's solvents to include the required one
                if electrolyte.solvents:
                    electrolyte.solvents[0] = solvent_to_include
                else:
                    electrolyte.solvents = [solvent_to_include]
                    electrolyte.stoich_solv = [1]
            
            # Try to include remaining cations
            if remaining_cations and remaining_systems <= len(remaining_cations):
                cation_to_include = list(remaining_cations)[0]
                cat_idx = cations_df[cations_df['formula'] == cation_to_include].index[0]
                cat_charge = cations_df.loc[cat_idx, 'charge']
                
                if electrolyte.cations:
                    electrolyte.cations[0] = cation_to_include
                    electrolyte.cat_charges[0] = cat_charge
                else:
                    electrolyte.cations = [cation_to_include]
                    electrolyte.cat_charges = [cat_charge]
                
                # Recalculate stoichiometry
                charges = electrolyte.cat_charges + electrolyte.an_charges
                electrolyte.stoich = electrolyte.charge_solver.solve_single_equation(charges)
            
            # Try to include remaining anions
            if remaining_anions and remaining_systems <= len(remaining_anions):
                anion_to_include = list(remaining_anions)[0]
                an_idx = anions_df[anions_df['formula'] == anion_to_include].index[0]
                an_charge = anions_df.loc[an_idx, 'charge']
                
                if electrolyte.anions:
                    electrolyte.anions[0] = anion_to_include
                    electrolyte.an_charges[0] = an_charge
                else:
                    electrolyte.anions = [anion_to_include]
                    electrolyte.an_charges = [an_charge]
                
                # Recalculate stoichiometry
                charges = electrolyte.cat_charges + electrolyte.an_charges
                electrolyte.stoich = electrolyte.charge_solver.solve_single_equation(charges)

        # Generate entries for different conditions
        if elyte_class == 'MS':
            # Only generate for minimum temperature for molten salt
            name = f'{elyte_class}-{i+1}-minT'
            entry = electrolyte.to_dict(name)
            elytes = pd.concat([elytes, pd.DataFrame([entry])], ignore_index=True)
        else:
            # Set distances and temperatures based on mode
            if is_rpmd:
                # RPMD mode: only low concentration and minimum temperature
                distances = [2.25]  # Only low concentration (2.25 nm)
                temperatures = [electrolyte.min_temp]  # Only minimum temperature
            else:
                # Non-RPMD mode: include both concentrations and temperatures
                distances = [1.75, 2.25]  # Both high (1.75 nm) and low (2.25 nm) concentrations
                temperatures = [electrolyte.min_temp, electrolyte.max_temp]  # Both min and max temperatures
            
            boxsize = 5  # nm
            minboxsize = 4  # nm
            minmol = 1
            
            for dis in distances:
                for temperature in temperatures:
                    conc = 0.62035049089/dis**3  # number per nm3
                    
                    # Calculate concentrations and box size
                    salt_molfrac = np.array(electrolyte.stoich)/sum(electrolyte.stoich)
                    mols = salt_molfrac/min(salt_molfrac)*minmol
                    salt_conc = salt_molfrac*conc/Electrolyte.Avog*1e24  # number per nm3
                    totalmol = np.sum(mols)
                    boxsize = (totalmol/conc)**(1/3)  # nm
                    
                    # Adjust minimum molecules if needed
                    newminmol = minmol
                    while minboxsize > boxsize:
                        newminmol += 1
                        mols = salt_molfrac/min(salt_molfrac)*newminmol
                        salt_conc = salt_molfrac*conc/Electrolyte.Avog*1e24
                        totalmol = np.sum(mols)
                        boxsize = (totalmol/conc)**(1/3)
                    
                    # Generate entry name with appropriate suffixes
                    name = f'{elyte_class}-{i+1}'
                    if temperature == electrolyte.min_temp:
                        name += '-minT'
                    else:
                        name += '-maxT'
                    
                    if dis == 2.25:
                        name += '-lowconc'
                    else:
                        name += '-highconc'
                    
                    # Create and add entry
                    entry = electrolyte.to_dict(name)
                    entry['temperature'] = temperature  # Set the actual temperature
                    elytes = pd.concat([elytes, pd.DataFrame([entry])], ignore_index=True)
    
    # Verify all required species were included (only for non-force mode)
    if random_only and not force_require:
        missing_species = {
            'solvents': required_species['solvents'] - included_species['solvents'],
            'cations': required_species['cations'] - included_species['cations'],
            'anions': required_species['anions'] - included_species['anions']
        }
        
        if any(missing_species.values()):
            missing_str = []
            for species_type, missing in missing_species.items():
                if missing:
                    missing_str.append(f"{species_type}: {', '.join(missing)}")
            raise RuntimeError(f"Failed to include all required species: {'; '.join(missing_str)}")

    # Save results if output file is specified
    if output_file:
        elytes.to_csv(output_file, index=False)
    
    return elytes

def main():
    """Main function to parse command line arguments and generate electrolytes."""
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            Script to generate a list of random electrolytes based on their classifications using an object-oriented approach.

            Classifications for doing non-RPMD simulations:
            (1) 40% Salt in protic solvents
            (2) 40% Salt in polar aprotic solvents
            (3) 10% Salt in ionic liquids
            (4) 5% Molten salt
            (5) 5% Aqueous electrolytes

            To change these ratios manually, go to Line 544-550.

            For RPMD simulations:
            (1) 30% Salt in protic solvents
            (2) 30% Salt in aprotic solvents
            (3) 40% Aqueous electrolytes

            If we turn on the flag, --random-only, then these ratios are ignored and we go with fully random mixtures.
            """),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rpmd', action='store_true', help='Generate electrolytes for RPMD simulations')
    parser.add_argument('--random-only', action='store_true', help='Generate only random electrolytes without type constraints')
    parser.add_argument('--cations', type=str, help='Path to cations CSV file', default='cations.csv')
    parser.add_argument('--anions', type=str, help='Path to anions CSV file', default='anions.csv')
    parser.add_argument('--solvents', type=str, help='Path to solvents CSV file', default='solvent.csv')
    parser.add_argument('--output', type=str, help='Path to output CSV file')
    parser.add_argument('--n-random', type=int, help='Number of random electrolytes to generate')
    parser.add_argument('--max-cations', type=int, default=4, help='Maximum number of cation species')
    parser.add_argument('--max-anions', type=int, default=4, help='Maximum number of anion species')
    parser.add_argument('--max-solvents', type=int, default=4, help='Maximum number of solvent species')
    parser.add_argument('--required-solvent', type=str, help='Comma-separated list of required solvent species')
    parser.add_argument('--required-cation', type=str, help='Comma-separated list of required cation species')
    parser.add_argument('--required-anion', type=str, help='Comma-separated list of required anion species')
    parser.add_argument('--force-require', action='store_true', 
                       help='Force at least one required species from each category in every generated system')
    args = parser.parse_args()
    
    # Prepare input files dictionary
    input_files = {
        'cations': args.cations,
        'anions': args.anions,
        'solvents': args.solvents
    }
    
    # Generate electrolytes with new force_require parameter
    generate_electrolytes(
        is_rpmd=args.rpmd,
        input_files=input_files,
        output_file=args.output,
        n_random=args.n_random,
        random_only=args.random_only,
        max_cations=args.max_cations,
        max_anions=args.max_anions,
        max_solvents=args.max_solvents,
        required_solvent=args.required_solvent,
        required_cation=args.required_cation,
        required_anion=args.required_anion,
        force_require=args.force_require
    )

if __name__ == "__main__":
    main()
