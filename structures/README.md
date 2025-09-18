# Protein Preparation

- pdb obtained from https://www.ebi.ac.uk/pdbe/entry/pdb/6o6f.
- Noted missing residues at the C terminus and affinity tag residues at the N terminus, along with a missing residue in the middle of the chain for chain B.
- Discarded chain A, keeping chain B with associated crystallographic waters and ligand. Removed the affinity tag residues (leaving the N terminus uncapped) and capped the C terminus as there were missing residues.
- Used [pdbfixer](https://github.com/openmm/pdbfixer) to add missing residues in the middle of the chain.
- Used [H++](http://newbiophysics.cs.vt.edu/H++/) to predict pkas and add protons.
- Used obabel to convert the ligand pdb to sdf and protonate at pH 7 (ligand is charged)
- Used [pdbfixer](https://github.com/openmm/pdbfixer) to solvated the protein (keeping crystalographic waters) and adding 0.15 M NaCl.
- Deleted a single Cl- to compensate for the negative charge of the ligand and ensure a neutral box.
- Ensured that there were no clashes with water when the ligand was added to create the complex.
