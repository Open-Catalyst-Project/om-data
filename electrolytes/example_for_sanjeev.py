from schrodinger.structure import StructureReader
from schrodinger.structutils import analyze
import json

# Read a structure
st = StructureReader.read("thing.mae")

# Read json metadat file
with open("thing.json") as f:
    metadata = json.load(f)

# Get a molecule number
res = next((res for res in st.residue if res.pdbres == "AAA "), None)
if res is None:
    mol_num = 1
else:
    mol_num = res.molecule_number

# Extract entire residues (can also say fillmol if you prefer) within 3 angstroms of molecule 1
ats = set(analyze.evaluate_asl(st, f"fillres within 3 mol {mol_num}"))
mol_included = [mol_num]


def get_extra_solutes(st, ats, radius, res_names, mol_included):
    """
    Gets the molecule numbers of solute molecules that are within the radius
    and not already included.
    """
    extras = []
    for at in ats:
        if st.atom[at].molecule_number in mol_included:
            continue
        if any(res_name == st.atom[at].getResidue().pdbres for res_name in res_names):
            extras.append(st.atom[at].molecule_number)
    return extras


extras = get_extra_solutes(st, ats, 3, ["AAA ", "BBB "], mol_included)
while extras:
    ats.update(
        analyze.evaluate_asl(
            st, f'fillres within 3 mol {",".join([str(i) for i in extras])}'
        )
    )
    mol_included.extend(extras)
    if len(ats) < 200:
        extras = get_extra_solutes(st, ats, 3, ["AAA ", "BBB "], mol_included)
    else:
        break
st2 = st.extract(sorted(ats), copy_props=True)
st2.write("thing2.mae")

# Now do a random water near the box edge, molecule 91
# label the molecule so we can keep track
for at in st.molecule[91].atom:
    at.property["i_m_label"] = 1
ats = analyze.evaluate_asl(st, "fillres within 3 mol 91")
st2 = st.extract(ats, copy_props=True)
st2.write("thing3.mae")

# Oh no! Our system is broken across the PBC boundary. Let's tell it to contract
# everthing to be centered on our molecule of interest (this will also handle if
# a molecule is split across a PBC)
from schrodinger.application.matsci import clusterstruct

mol_ats = [at.index for at in st2.atom if at.property.pop("i_m_label", None) == 1]
clusterstruct.contract_structure2(st2, contract_on_atoms=mol_ats)
st2.write("thing3b.mae")

# Suppose we have several structures and we want to group them by conformers
from schrodinger.comparison import are_conformers
from schrodinger.application.jaguar.utils import group_items


def get_env_of_mol(st, mol_idx, radius):
    st_copy = st.copy()
    for at in st_copy.molecule[mol_idx].atom:
        at.property["i_m_label"] = 1
    ats = analyze.evaluate_asl(st_copy, f"fillres within {radius} mol {mol_idx}")
    st2 = st_copy.extract(ats, copy_props=True)
    mol_ats = [at.index for at in st2.atom if at.property.pop("i_m_label", None) == 1]
    clusterstruct.contract_structure2(st2, contract_on_atoms=mol_ats)
    return st2


st_list = [get_env_of_mol(st, i, 3) for i in range(2, 100)]

grouped_items = group_items(st_list, are_conformers)
# we now have lists of lists of structures where each list of structures are conformers of each other

from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper
from schrodinger.structutils import rmsd


def renumber_molecules_to_match(mol_list):
    """
    Ensure that topologically equivalent sites are equivalently numbered
    """
    mapper = ConnectivityAtomMapper(use_chirality=False)
    atlist = range(1, mol_list[0].atom_total + 1)
    renumbered_mols = [mol_list[0]]
    for mol in mol_list[1:]:
        _, r_mol = mapper.reorder_structures(mol_list[0], atlist, mol, atlist)
        renumbered_mols.append(r_mol)
    return renumbered_mols


# Now we have ensured that topologically related atoms are equivalently numbered (up to molecular symmetry)
grouped_items = [renumber_molecules_to_match(items) for items in grouped_items]

# example grouping
group = grouped_items[1]
# Compare all atoms for RMSD (remember that Schrodinger atoms/molecules  are 1-indexed
at_list = list(range(1, group[0].atom_total + 1))
# Note that this function will translate/rotate all structures to align with the first one
# If this is not desired, run function on a copy, e.g. other.copy()
rmsds = [
    rmsd.superimpose(group[0], at_list, other, at_list, use_symmetry=True)
    for other in group[1:]
]
