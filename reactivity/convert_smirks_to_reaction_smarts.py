from rdkit import Chem
from rdkit.Chem import AllChem
from copy import copy
import re
import itertools

smirks = 'CC(C)(C[O:11][N+:10]([O-])=O)C.[Ar]>>CC(C)(C[O:11])C.[O-][N+:10]=O.[Ar] 10,11-10;10,11-11'
smirks = 'C1=C[C:63]2=[C:64](C=C1)[CH:73]=[CH:72][C:61]([S-:10])=[CH:62]2.Cl[C:34]1=[CH:33][CH:32]=[C:31]([CH+:20][C:41]2=[CH:42][CH:43]=[C:44](Cl)[CH:45]=[CH:46]2)[CH:36]=[CH:35]1>>Cl[C:34]1=[CH:33][CH:32]=[C:31]([CH:20]([S:10][C:61]2=[CH:62][C:63]3=[C:64](C=CC=C3)[CH:73]=[CH:72]2)[C:41]2=[CH:42][CH:43]=[C:44](Cl)[CH:45]=[CH:46]2)[CH:36]=[CH:35]1 10=20'

def convert_smirks_to_reaction_smarts(smirks):
    # Remove H labels on mapped atoms to make SMARTS more promiscuous
    def delgroup1(m):
        return m.group(0).replace(m.group(1),'')
    smirks = re.sub(r'\[.*[A-Zaz](H\d*).*\]', delgroup1, smirks)

    rxn = AllChem.ReactionFromSmarts(smirks)
    # Kekulize reactants and products
    AllChem.SanitizeRxnAsMols(rxn)
    # Round-trip the reaction to make SMARTS patterns include aromaticity in atom labels
    rxn = AllChem.ReactionFromSmarts(AllChem.ReactionToSmiles(rxn), useSmiles=True)

    # Remove the unmapped atoms
    new_rs = []
    for r in rxn.GetReactants():
        for at in r.GetAtoms():
            if not at.GetAtomMapNum() or at.GetSymbol() in {'Ar'}:
                at.SetAtomicNum(0)
        new_rs.append(Chem.DeleteSubstructs(r, Chem.MolFromSmarts('[#0]')))
    new_ps = []
    for r in rxn.GetProducts():
        for at in r.GetAtoms():
            if not at.GetAtomMapNum() or at.GetSymbol() in {'Ar'}:
                at.SetAtomicNum(0)
        new_ps.append(Chem.DeleteSubstructs(r, Chem.MolFromSmarts('[#0]')))

    r_smarts = [Chem.MolToSmarts(r) for r in new_rs]
    r_smarts = sorted([r for r in r_smarts if r])
    p_smarts = [Chem.MolToSmarts(p) for p in new_ps]
    p_smarts = sorted([p for p in p_smarts if p])
    new_reactions = {}
    # Do both the forward and back reaction
    for r, p in itertools.permutations([r_smarts, p_smarts]):
        new_smarts = '.'.join(r) + '>>' + '.'.join(p)
        rxn = AllChem.ReactionFromSmarts(new_smarts, useSmiles=True)
        rxn_copy = copy(rxn)
        AllChem.RemoveMappingNumbersFromReactions(rxn_copy)
        rxn_key = AllChem.ReactionToSmarts(rxn_copy)
        new_reactions[rxn_key] = rxn
    return new_reactions
new_reactions = convert_smirks_to_reaction_smarts(smirks)
for key, val in new_reactions.items():
    print(key)
    print(AllChem.ReactionToSmarts(val))
