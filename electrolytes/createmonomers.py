import os
from schrodinger.structure import StructureReader, StructureWriter
from schrodinger.application.jaguar.utils import mmjag_update_lewis
from schrodinger.structutils.analyze import evaluate_asl
import string

def generate_molres(length):
    molres = []
    alphabet = string.ascii_uppercase
    num_alphabet = len(alphabet)
    
    for i in range(length):
        if i < num_alphabet:
            letter = alphabet[i]
            molres.append(letter * 3)
        else:
            number = i - num_alphabet + 1
            molres.append(str(number) * 3)
    
    return molres

def write_monomers(cat, an, solv, charges, directory):
    st_list = []
    species = cat+an+solv
    molres = generate_molres(len(cat+an+solv))
    chainIDs = len(cat + an)*['A']+len(solv)*['B']
    for sp, charge, res_name, chain_name in zip(species, charges, molres, chainIDs):
        fname = sp+'.pdb'
        if sp in {'C9H18NO', 'AsF6-', 'OV+2', 'O2V+'}:
            # Special treatment:
            # TEMPO: Desmond does not like radicals
            # As-F, OV+2, O2V+: No parameters
            fname = sp + '.mae'
        print(fname)
        st = StructureReader.read(os.path.join('ff', fname))
        if fname.endswith('.pdb'):
            st.property['i_m_Molecular_charge'] = charge
            mmjag_update_lewis(st)
        # Iterate over residues
        for res in st.residue:
            res.chain = chain_name
            res.pdbres = res_name
            print(res, res.pdbres)
        st_list.append(st)
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
