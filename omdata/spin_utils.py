import mendeleev
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structure import Structure

## NB: This function must be run inside a Schrodinger virtual environment with the mendeleev package pip-installed

def guess_spin_state(st: Structure) -> int:
    """
    Guess a realistic spin state for a system which may have any number of metals present

    :param st: The structure to guess the spin state of
    :return: Guess spin multiplicity (i.e. 1 for singlet)
    """
    metals = evaluate_asl(st, "metals")
    # We will assume antiferromagnetic coupling for multimetallic systems
    # to ensure we don't put the spin state outside our acceptable range
    total_spin = 0
    for idx, metal_idx in enumerate(metals):
        metal_at = st.atom[metal_idx]
        local_spin = mendeleev.element(metal_at.element).ec.ionize(metal_at.formal_charge).unpaired_electrons()
        # Assume 2nd, 3rd row TMs are low spin, Ln are high spin
        if metal_at.atomic_number > 36 and not metal_at.atomic_number in range(59,70): 
            local_spin = local_spin % 2 
        total_spin += (-1) ** idx * local_spin
    return abs(total_spin) + 1
