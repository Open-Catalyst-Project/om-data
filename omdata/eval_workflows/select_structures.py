from fairchem.core.datasets import AseDBDataset
import random
import argparse
import os
from monty.serialization import dumpfn
from architector.io_molecule import convert_io_molecule
import mendeleev

random.seed(42)


open_shell_metals = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"]

def num_open_shell_metals(atoms):
    return len([atom for atom in atoms if atom.symbol in open_shell_metals])


def ie_ea_processing(atoms):
    structs_to_add = []
    add_electron_charge = atoms.info["charge"] - 1
    remove_electron_charge = atoms.info["charge"] + 1
    if atoms.info["spin"] == 1:
        add_electron_spin_multiplicity = [2]
        remove_electron_spin_multiplicity = [2]
    else:
        add_electron_spin_multiplicity = [atoms.info["spin"]+1, atoms.info["spin"]-1]
        remove_electron_spin_multiplicity = [atoms.info["spin"]+1, atoms.info["spin"]-1]

    for spin_multiplicity in add_electron_spin_multiplicity:
        structs_to_add.append({
            "charge": add_electron_charge,
            "spin": spin_multiplicity,
            "source": atoms.info["source"]
        })

    for spin_multiplicity in remove_electron_spin_multiplicity:
        structs_to_add.append({
            "charge": remove_electron_charge,
            "spin": spin_multiplicity,
            "source": atoms.info["source"]
        })

    return structs_to_add


def sample_structures_for_unoptimized_ie_ea(structures, number, idx_name):
    sampled_structures = []
    viable_idxs = []
    for idx in range(len(structures)):
        if idx_name == "e_idx":
            if num_open_shell_metals(structures.get_atoms(idx)) > 2:
                continue
            viable_idxs.append(idx)
        elif idx_name == "mca_idx":
            viable_idxs.append(idx)
        elif idx_name == "mcc_idx":
            if len(structures[idx].info["metal_list"]) > 1:
                continue
            viable_idxs.append(idx)
    for idx in random.sample(viable_idxs, number):
        if idx_name in ["e_idx", "mca_idx"]:
            structs_to_add = ie_ea_processing(structures.get_atoms(idx))
        elif idx_name == "mcc_idx":
            structs_to_add = ie_ea_processing(structures[idx])
        for struct in structs_to_add:
            struct[idx_name] = idx
            sampled_structures.append(struct)
    return sampled_structures


def sample_structures_for_unoptimized_spin_gap(structures, number, idx_name):
    cached_unpaired_electrons = {}
    sampled_structures = []
    idx_list = list(range(len(structures)))
    random.shuffle(idx_list)
    samples_taken = 0
    for idx in idx_list:
        new_spins = []
        if idx_name in ["e_idx", "mca_idx"]:
            tmp_atoms = structures.get_atoms(idx)
            if tmp_atoms.info["spin"] > 2:
                if idx_name == "e_idx":
                    if num_open_shell_metals(tmp_atoms) > 1:
                        continue
                new_spin = tmp_atoms.info["spin"] - 2
                while new_spin > 0:
                    new_spins.append(new_spin)
                    new_spin -= 2
        elif idx_name in ["mcc_idx"]:
            tmp_atoms = structures[idx]
            if len(tmp_atoms.info["metal_list"]) > 2 and idx_name == "mcc_idx":
                continue
            max_num_unpaired_electrons = 0
            lanthanide_present = False
            for metal in tmp_atoms.info["metal_list"]:
                if metal[0] in ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm"]:
                    lanthanide_present = True
                if (metal[0], metal[1]) not in cached_unpaired_electrons:
                    cached_unpaired_electrons[(metal[0], metal[1])] = mendeleev.element(metal[0]).ec.ionize(metal[1]).unpaired_electrons()
                max_num_unpaired_electrons += cached_unpaired_electrons[(metal[0], metal[1])]
            if max_num_unpaired_electrons != 0:
                new_spin = max_num_unpaired_electrons+1
                while new_spin > 0:
                    if new_spin != tmp_atoms.info["spin"]:
                        if new_spin < 12 or lanthanide_present:
                            new_spins.append(new_spin)
                    new_spin -= 2

        if new_spins != []:
            samples_taken += 1
            for new_spin in new_spins:
                assert new_spin != tmp_atoms.info["spin"]
                assert new_spin%2 == tmp_atoms.info["spin"]%2
                sampled_structures.append({
                    "charge": tmp_atoms.info["charge"],
                    "spin": new_spin,
                    "source": tmp_atoms.info["source"],
                    idx_name: idx,
                })
            if samples_taken == number:
                break
    print(samples_taken, "samples taken for unoptimized spin gap for", idx_name)
    return sampled_structures


def main(args):
    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    elytes = AseDBDataset({"src": input_path + "/test_elytes_samples"})
    metal_complexes_arch = AseDBDataset({"src": input_path + "/test_organometallics_samples"})

    metal_complexes_cod = []
    for filename in os.listdir(input_path + "/cod_complexes"):
        atoms = convert_io_molecule(input_path + "/cod_complexes/" + filename).ase_atoms
        atoms.info["charge"] = int(filename[:-4].split("_")[-2])
        atoms.info["spin"] = int(filename[:-4].split("_")[-1])
        atoms.info["source"] = filename
        metal_list = []
        with open(input_path + "/cod_complexes/" + filename, "r") as f:
            for line in f.readlines():
                split_line = line.split()
                if len(split_line) > 7:
                    if split_line[3] in open_shell_metals:
                        if len(split_line) > 8:
                            for kk in range(8, len(split_line)):
                                if split_line[kk][0:4] == "CHG=":
                                    metal_list.append([split_line[3], int(split_line[kk][4:])])
                                    break
                            else:
                                metal_list.append([split_line[3], 0])
                        else:
                            metal_list.append([split_line[3], 0])
        negative_or_zero_ox_found = False
        for metal in metal_list:
            if metal[1] <= 0:
                negative_or_zero_ox_found = True
                break
        if not negative_or_zero_ox_found:
            atoms.info["metal_list"] = metal_list
            metal_complexes_cod.append(atoms)

    unoptimized_ie_ea_structures_elytes = sample_structures_for_unoptimized_ie_ea(elytes, 4000, "e_idx")
    print(len(unoptimized_ie_ea_structures_elytes), "elyte calcs to run for unoptimized IE/EA")
    unoptimized_ie_ea_structures_metal_complexes_arch = sample_structures_for_unoptimized_ie_ea(metal_complexes_arch, 1000, "mca_idx")
    print(len(unoptimized_ie_ea_structures_metal_complexes_arch), "Architector metal complex calcs to run for unoptimized IE/EA")
    unoptimized_ie_ea_structures_metal_complexes_cod = sample_structures_for_unoptimized_ie_ea(metal_complexes_cod, 4000, "mcc_idx")
    print(len(unoptimized_ie_ea_structures_metal_complexes_cod), "COD metal complex calcs to run for unoptimized IE/EA")

    dumpfn(unoptimized_ie_ea_structures_elytes, os.path.join(output_path, "unoptimized_ie_ea_structures_elytes.json"))
    dumpfn(unoptimized_ie_ea_structures_metal_complexes_arch, os.path.join(output_path, "unoptimized_ie_ea_structures_metal_complexes_arch.json"))
    dumpfn(unoptimized_ie_ea_structures_metal_complexes_cod, os.path.join(output_path, "unoptimized_ie_ea_structures_metal_complexes_cod.json"))

    unoptimized_spin_gap_structures_elytes = sample_structures_for_unoptimized_spin_gap(elytes, 1000, "e_idx")
    print(len(unoptimized_spin_gap_structures_elytes), "elyte calcs to run for unoptimized spin gap")
    unoptimized_spin_gap_structures_metal_complexes_arch = sample_structures_for_unoptimized_spin_gap(metal_complexes_arch, 3000, "mca_idx")
    print(len(unoptimized_spin_gap_structures_metal_complexes_arch), "Architector metal complex calcs to run for unoptimized spin gap")
    unoptimized_spin_gap_structures_metal_complexes_cod = sample_structures_for_unoptimized_spin_gap(metal_complexes_cod, 5000, "mcc_idx")
    print(len(unoptimized_spin_gap_structures_metal_complexes_cod), "COD metal complex calcs to run for unoptimized spin gap")

    dumpfn(unoptimized_spin_gap_structures_elytes, os.path.join(output_path, "unoptimized_spin_gap_structures_elytes.json"))
    dumpfn(unoptimized_spin_gap_structures_metal_complexes_arch, os.path.join(output_path, "unoptimized_spin_gap_structures_metal_complexes_arch.json"))
    dumpfn(unoptimized_spin_gap_structures_metal_complexes_cod, os.path.join(output_path, "unoptimized_spin_gap_structures_metal_complexes_cod.json"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=".")
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)