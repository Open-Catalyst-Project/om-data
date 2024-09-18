import os
from schrodinger.structure import StructureReader, StructureWriter
from schrodinger.application.jaguar.utils import mmjag_update_lewis
from schrodinger.structutils.analyze import evaluate_asl

def write_monomers(species, charges, directory):
    st_list = []
    for sp, charge in zip(species, charges):
        print(sp+'.pdb')
        for st in StructureReader(os.path.join('ff', sp+".pdb")):
            st.property['i_m_Molecular_charge'] = charge
            mmjag_update_lewis(st)
            zob_metals(st)
            st_list.append(st)
    print(st_list)
    with StructureWriter(os.path.join(directory, 'monomers.maegz')) as writer:
         writer.extend(st_list)

def zob_metals(st):
    """
    Make bonds to metals zero-order bonds and collect charge onto the metal center.

    This only applies to VO2^+ and VO^2+
    """
    metals = evaluate_asl(st, "metals")
    if metals:
        assert len(metals) == 1
        charge = st.formal_charge
        for at in st.atom:
            at.formal_charge = 0
        for at_idx in metals:
            for bond in st.atom[at_idx].bond:
                bond.order = 0
            st.atom[at_idx].formal_charge = charge
