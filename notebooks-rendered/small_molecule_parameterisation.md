# Parameterising Small Molecules with OpenFF

This is the first of two jupyter notebooks on handling force fields using [Open Force Field's](https://openforcefield.org/) software, and subsequent molecular dynamics and analysis. This notebook describes the parameterisation of small molecules, while the second notebook (`protein_ligand_complex_parameterisation_and_md.ipynb`) will take you through parameterising a protein-ligand complex, running molecular dynamics, and performing some analysis of pose stability and interactions.

### Prerequisites

 - Basic knowledge of Python
 - Basic familiarity with molecular mechanics force fields and molecular dynamics simulations (see talk by Danny Cole)

### The Plan

| Action | Software|
|--|--|
| [Load and inspect a force field](#loading_ff) | OpenFF Toolkit
| [Create a representation of your chemical system](#topology) | OpenFF Toolkit
| [Parameterise your system and run a quick simulation](#interchange) | OpenFF Interchange, OpenMM
| [Rapidly assign partial charges with a graph neural network model](#gnn_charges) | OpenFF Toolkit, OpenFF NAGL Models
| [Review what you've learnt](#summary) | 
| [Check out other OpenFF tutorials](#further_materials) | 
 
### Jupyter Cheat Sheet

- To run the currently highlighted cell and move focus to the next cell, hold <kbd>&#x21E7; Shift</kbd> and press <kbd>&#x23ce; Enter</kbd>;
- To run the currently highlighted cell and keep focus in the same cell, hold <kbd>&#x21E7; Ctrl</kbd> and press <kbd>&#x23ce; Enter</kbd>;
- To get help for a specific function, place the cursor within the function's brackets, hold <kbd>&#x21E7; Shift</kbd>, and press <kbd>&#x21E5; Tab</kbd>;

### Acknowledgements

Most of this material was adapted from the [2023 CCPBioSim Workshop Open Force Field Sessions](https://github.com/openforcefield/ccpbiosim-2023?tab=readme-ov-file) created by Matt Thompson and Jeff Wagner.

### Maintainers
 - Finlay Clark -- finlay.clark@newcastle.ac.uk (@fjclark)



<a id="loading_ff"></a>
## 1. Force fields are specified in `.offxml` files and can be loaded with the `ForceField` class

OpenFF's force fields use the The SMIRKS Native Open Force Field (SMIRNOFF) [specification](https://openforcefield.github.io/standards/standards/smirnoff/) and are conventionally encoded in `.offxml` files. The spec fully describes the contents of a SMIRNOFF force field, how parameters should be applied, and several other important usage details. You could implement a SMIRNOFF engine in your own code, but conveniently the OpenFF Toolkit already provides this and a handful of utilities. Let's load up the latest OpenFF small molecule force field, OpenFF 2.2.1, and inspect its contents! This force field shares the code name "Sage" with all other force fields with the same major version number (2.x.x).


```python
from openff.toolkit import ForceField

sage = ForceField("openff-2.2.1.offxml")
sage
```




    <openff.toolkit.typing.engines.smirnoff.forcefield.ForceField at 0x7f2f3a070da0>



If you'd like to see the raw file on disk that's being parsed, [here's the file on GitHub](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff-2.2.1.offxml).

Each section of a force field is stored in memory within `ParameterHandler` objects, which can be looked up with brackets (just like looking up values in a dictionary):


```python
print(sage.registered_parameter_handlers)

vdw_handler = sage["vdW"]
vdw_handler
```

    ['Constraints', 'Bonds', 'Angles', 'ProperTorsions', 'ImproperTorsions', 'vdW', 'Electrostatics', 'LibraryCharges', 'ToolkitAM1BCC']





    <openff.toolkit.typing.engines.smirnoff.parameters.vdWHandler at 0x7f2f381b4cb0>



Each `ParameterHandler` in turn stores a list of parameters in its `.parameters` attribute, in addition to some information specific to its portion of the potential energy function:


```python
print(f"vdw_handler cutoff: {vdw_handler.cutoff}")
print(f"vdw_handler combining rules: {vdw_handler.combining_rules}")
print(f"vdw_handler scale14: {vdw_handler.scale14}")
print(f"vdw_handler parameters: {vdw_handler.parameters}")
```

    vdw_handler cutoff: 9.0 angstrom
    vdw_handler combining rules: Lorentz-Berthelot
    vdw_handler scale14: 0.5
    vdw_handler parameters: [<vdWType with smirks: [#1:1]  epsilon: 0.0157 kilocalorie / mole  id: n1  rmin_half: 0.6 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4]  epsilon: 0.01577948280971 kilocalorie / mole  id: n2  rmin_half: 1.48419980825 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]  epsilon: 0.01640924602775 kilocalorie / mole  id: n3  rmin_half: 1.449786411317 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]  epsilon: 0.0157 kilocalorie / mole  id: n4  rmin_half: 1.287 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]  epsilon: 0.0157 kilocalorie / mole  id: n5  rmin_half: 1.187 angstrom  >, <vdWType with smirks: [#1:1]-[#6X4]~[*+1,*+2]  epsilon: 0.0157 kilocalorie / mole  id: n6  rmin_half: 1.1 angstrom  >, <vdWType with smirks: [#1:1]-[#6X3]  epsilon: 0.01561134320353 kilocalorie / mole  id: n7  rmin_half: 1.443812569645 angstrom  >, <vdWType with smirks: [#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]  epsilon: 0.01310699839698 kilocalorie / mole  id: n8  rmin_half: 1.377051329051 angstrom  >, <vdWType with smirks: [#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]  epsilon: 0.01479744504464 kilocalorie / mole  id: n9  rmin_half: 1.370482808197 angstrom  >, <vdWType with smirks: [#1:1]-[#6X2]  epsilon: 0.015 kilocalorie / mole  id: n10  rmin_half: 1.459 angstrom  >, <vdWType with smirks: [#1:1]-[#7]  epsilon: 0.01409081474669 kilocalorie / mole  id: n11  rmin_half: 0.6192778454102 angstrom  >, <vdWType with smirks: [#1:1]-[#8]  epsilon: 1.232599966667e-05 kilocalorie / mole  id: n12  rmin_half: 0.2999999999997 angstrom  >, <vdWType with smirks: [#1:1]-[#16]  epsilon: 0.0157 kilocalorie / mole  id: n13  rmin_half: 0.6 angstrom  >, <vdWType with smirks: [#6:1]  epsilon: 0.0868793154488 kilocalorie / mole  id: n14  rmin_half: 1.953447017081 angstrom  >, <vdWType with smirks: [#6X2:1]  epsilon: 0.21 kilocalorie / mole  id: n15  rmin_half: 1.908 angstrom  >, <vdWType with smirks: [#6X4:1]  epsilon: 0.1088406109251 kilocalorie / mole  id: n16  rmin_half: 1.896698071741 angstrom  >, <vdWType with smirks: [#8:1]  epsilon: 0.2102061007896 kilocalorie / mole  id: n17  rmin_half: 1.706036917087 angstrom  >, <vdWType with smirks: [#8X2H0+0:1]  epsilon: 0.1684651402602 kilocalorie / mole  id: n18  rmin_half: 1.697783613804 angstrom  >, <vdWType with smirks: [#8X2H1+0:1]  epsilon: 0.2094735324129 kilocalorie / mole  id: n19  rmin_half: 1.682099169199 angstrom  >, <vdWType with smirks: [#7:1]  epsilon: 0.1676915150424 kilocalorie / mole  id: n20  rmin_half: 1.799798315098 angstrom  >, <vdWType with smirks: [#16:1]  epsilon: 0.25 kilocalorie / mole  id: n21  rmin_half: 2.0 angstrom  >, <vdWType with smirks: [#15:1]  epsilon: 0.2 kilocalorie / mole  id: n22  rmin_half: 2.1 angstrom  >, <vdWType with smirks: [#9:1]  epsilon: 0.061 kilocalorie / mole  id: n23  rmin_half: 1.75 angstrom  >, <vdWType with smirks: [#17:1]  epsilon: 0.2656001046527 kilocalorie / mole  id: n24  rmin_half: 1.85628721824 angstrom  >, <vdWType with smirks: [#35:1]  epsilon: 0.3218986365974 kilocalorie / mole  id: n25  rmin_half: 1.969806594135 angstrom  >, <vdWType with smirks: [#53:1]  epsilon: 0.4 kilocalorie / mole  id: n26  rmin_half: 2.35 angstrom  >, <vdWType with smirks: [#3+1:1]  epsilon: 0.0279896 kilocalorie / mole  id: n27  rmin_half: 1.025 angstrom  >, <vdWType with smirks: [#11+1:1]  epsilon: 0.0874393 kilocalorie / mole  id: n28  rmin_half: 1.369 angstrom  >, <vdWType with smirks: [#19+1:1]  epsilon: 0.1936829 kilocalorie / mole  id: n29  rmin_half: 1.705 angstrom  >, <vdWType with smirks: [#37+1:1]  epsilon: 0.3278219 kilocalorie / mole  id: n30  rmin_half: 1.813 angstrom  >, <vdWType with smirks: [#55+1:1]  epsilon: 0.4065394 kilocalorie / mole  id: n31  rmin_half: 1.976 angstrom  >, <vdWType with smirks: [#9X0-1:1]  epsilon: 0.003364 kilocalorie / mole  id: n32  rmin_half: 2.303 angstrom  >, <vdWType with smirks: [#17X0-1:1]  epsilon: 0.035591 kilocalorie / mole  id: n33  rmin_half: 2.513 angstrom  >, <vdWType with smirks: [#35X0-1:1]  epsilon: 0.0586554 kilocalorie / mole  id: n34  rmin_half: 2.608 angstrom  >, <vdWType with smirks: [#53X0-1:1]  epsilon: 0.0536816 kilocalorie / mole  id: n35  rmin_half: 2.86 angstrom  >, <vdWType with smirks: [#1]-[#8X2H2+0:1]-[#1]  epsilon: 0.1521 kilocalorie / mole  id: n-tip3p-O  sigma: 3.1507 angstrom  >, <vdWType with smirks: [#1:1]-[#8X2H2+0]-[#1]  epsilon: 0.0 kilocalorie / mole  id: n-tip3p-H  sigma: 1 angstrom  >, <vdWType with smirks: [#54:1]  epsilon: 0.561 kilocalorie / mole  id: n36  sigma: 4.363 angstrom  >]


From here you can inspect all the way down to individual parameters, which are stored in custom objects (in this case, `vdWType`). Let's look at the type with id `n16`, which looks like a generic carbon with four bonded neighbors:


```python
vdw_type = vdw_handler.parameters[15]
vdw_type
```




    <vdWType with smirks: [#6X4:1]  epsilon: 0.1088406109251 kilocalorie / mole  id: n16  rmin_half: 1.896698071741 angstrom  >



Note that the type contains both the physical parameters (sigma and epsilon, for a conventional 12-6 Lennard-Jones potential), but also an associated SMIRKS pattern. This particular SMIRKS pattern is fairly simple, but some can get much more complex.

The toolkit uses these SMIRKS patterns and direct chemical perception to assign parameters to particular atoms (or bonds, angles, etc.).

We'll use OpenFF 2.2.1 for the remainder of this tutorial, but you can learn more about this and other SMIRNOFF force fields below:
<details>
  <summary><b>Click here to learn about available and planned SMIRNOFF force fields</b></summary>

# Existing force fields

## From OpenFF

### smirnoff99Frosst

This [force field](https://github.com/openforcefield/smirnoff99Frosst) is mostly a historical artifact today. It is the first SMIRNOFF force field, dating back to 2016. It is based on Merck-Frosst's [parm@frosst](http://www.ccl.net/cca/data/parm_at_Frosst/) and an old AMBER force field, parm99, which predates GAFF.

It is not recommended for general use today, but you might see it in papers that compare the performance of different force fields.

### Parsley

The Parsley line of force fields (`openff-1.y.z.offxml`) was OpenFF's [first full force field release](https://openforcefield.org/community/news/general/introducing-openforcefield-1.0/). Based on `smirnoff99Frosst`, these force fields are primarily re-fits of valence parameters using a large number of QM structures pulled from QCArchive. The first version was 1.0.0 and subsequent re-fits produced versions 1.1.0, 1.2.0, and 1.3.0. More detail is provided in an [associated paper](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00571).

### Sage

The Sage line of force fields (`openff-2.y.z.offxml`) continued the process of fitting to more (and more diverse) QM datasets, but also included a re-fit of the Lennard-Jones parameters. Small molecule geometries and energies [improved, in general,](https://openforcefield.org/community/news/general/sage2.0.0-release/) significantly over Parsley. These improvements notably transferred to protein-ligand binding free energies despite Sage not being specifically fit to them. For more, see the [associated paper](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00039).

[Subsequent releases](https://github.com/openforcefield/openff-forcefields/releases) used different fitting procedures and tweaks to parameter typing to improve performance and address issues with several specific chemistries. The latest release, **Sage 2.2.1 (`openff-2.2.1.offxml`) is the recommended force field for small molecule studies.**

## Ports

### Water models

OpenFF has ported [several existing water models](https://github.com/openforcefield/openff-forcefields/blob/main/docs/water-models.md) to SMIRNOFF format, including:

- TIP3P
- TIP3P-FB
- TIP4P-FB
- OPC
- OPC3

Existing main-line OpenFF force fields are fit against TIP3P water, so use of others is not (currently) recommended. This might change in the future, or OpenFF might even fit a new water model in a future release.

## ff14SB

OpenFF, in collaboration with Dave Cerutti of the Amber community, created a port of [ff14SB](https://pubs.acs.org/doi/10.1021/acs.jctc.5b00255), a popular Amber protein force field. There are some small numerical differences with how improper torsions are evaluated, but all other terms reproduce a canonical Amber source to high accuracy. **This is the only protein force field currently in SMIRONOFF (`.offxml`) format** and therefore the current recommendation for use with proteins. Primarily for technical reasons, porting other Amber force fields is not planned.

## Non-main-line force fields

### SMIRNOFF plugins

https://github.com/openforcefield/smirnoff-plugins

https://github.com/jthorton/de-forcefields

# Future force fields

## From OpenFF

### Rosemary
A future line of force fields from OpenFF (code name "Rosemary", starting with `openff-3.0.0.offxml`) is intended to handle small molecules and biopolymers in a _self-consistent_ manner. The first release is expected to handle proteins, but future versions  may cover nucleic acids. The performance, depending on the metrics used, is hoped to be comparable with existing Amber-family protein force fields.

There is no specific release date planned for Rosemary, but it may be available in 2026 (a beta release candidate may also be publically available prior to the full release).

### Graph net charge assignment

TODO: UPDAATE AND MENTION 2.3

The Sage 2.3.0 release is expected imminently and will include graph-convolutional neutral network (GCNN)-based charge assignment using [NAGL](https://github.com/openforcefield/openff-nagl) by default. The charge model is trained to reproduce AM1-BCC charges without the typical $O(N^3)$ scaling, making it suitable for large (>> 100 atoms) molecules). The [second release candidate](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff-2.3.0-rc2.offxml) (which may or may not become the final version) is already available for you to try!

### Virtual sites

Another release from OpenFF may include some virtual site parameters with off-center charges. No release date is planned, but the most of the supporting infrastructure is currently in place and some early studies have shown promise for better representing electrostatics of chemistries such as halogens and aromatic nitrogens.

## From you!

Anybody can write a SMIRNOFF force field! This workshop doesn't have time to cover force field _fitting_, but there are plenty of freely-available tools used today that can re-fit existing force fields or generate something new from the ground up. Once you've fit a new force field, a small Python package can distribute it in a way that the toolkit can [automatically load](https://docs.openforcefield.org/projects/toolkit/en/stable/faq.html#how-can-i-distribute-my-own-force-fields-in-smirnoff-format)!
</details>

<a id="topology"></a>
## 2. The `Topology` class represents a chemical system containing one or more `Molecule`s

Now we've loaded our desired force field (OpenFF 2.2.1), we need to specify the chemical system we want to assign force field parameters to ("parameterise"). Our system will be represented by a `Topology`, which we will build from one or more `Molecule`s. 

As a simple example, let's build a `Topology` containing an small molecule with some features which illustrate how parameters are applied according to SMIRKS matches. We'll use the crotonate anion, but you could draw any molecule you like and convert it to a SMILES string using tools like ChemDraw and [MolView](https://molview.org/).


```python
from openff.toolkit import Molecule, Topology

molecule = Molecule.from_smiles("C/C=C/C(=O)[O-]")
molecule
```


    


    /opt/conda/envs/openff-env/lib/python3.12/site-packages/nglview/__init__.py:12: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      import pkg_resources



    
![svg](output_12_2.svg)
    


We can also visualise our molecule in 3D using NGLView, but only after generating 3D coordinates with `generate_conformers`:


```python
molecule.generate_conformers(n_conformers=1)
molecule.visualize(backend="nglview")
```


    NGLWidget()


<div class="alert alert-warning" style="max-width: 700px; margin-left: auto; margin-right: auto;">
    ⚠️ Be careful when creating molecules from SMILES with undefined stereochemistry. By default, an `UndefinedStereochemistryError` will be raised, but this can be downgraded to a warning by setting the <code>allow_undefined_stereo=True</code>. This will create a molecule with undefined stereochemistry, which might lead to incorrect parameterisation or surprising conformer generation. See the <a href="https://docs.openforcefield.org/en/latest/faq.html">"I'm getting stereochemistry errors when loading a molecule from a SMILES string" FAQ</a> for more details.
</div>

Topologies can always be assembled by constructing individual molecules and adding them together; these methods are for making common operations easier.

To convert a single `Molecule` to a `Topology`, you can use either `Molecule.to_topology()` or `Topology.from_molecules`


```python
topology = molecule.to_topology()

# Or, equivalently:
topology = Topology.from_molecules(molecules=[molecule])
```

From here we can add as many other molecules as we wish. For example, we can create a water molecule and add it to a topology 100 times.


```python
water = Molecule.from_smiles("O")
topology_with_water = Topology(topology)

for index in range(100):
    topology_with_water.add_molecule(water)

topology_with_water.n_molecules
```




    101



<div class="alert alert-info" style="max-width: 500px; margin-left: auto; margin-right: auto;">
    ℹ️ Positions are <i>optional</i> in <code>Molecule</code> (any by extension <code>Topology</code>) objects, so visualizing this topology in 3D doesn't make sense. Using it in a simulation would requiring assigning positions using a tool like Packmol or PDBFixer. Running simulations will be discussed later.
</div>

Keeping in mind that topologies are just collections of molecules, we can look up individual molecules by index in the `Topology.molecule()` function.


```python
topology_with_water.molecule(0), topology_with_water.molecule(1), topology_with_water.molecule(-1)
```




    (Molecule with name '' and SMILES '[H]/[C]([C](=[O])[O-])=[C](/[H])[C]([H])([H])[H]',
     Molecule with name '' and SMILES '[H][O][H]',
     Molecule with name '' and SMILES '[H][O][H]')



<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Build a <code>Topology</code> containing an MCL-1 ligand. Create the <code>Molecule</code> from an SDF file  (take a look at the docstring of <code>Molecule</code> to see how this can be done). Also, see <a href="https://docs.openforcefield.org/projects/toolkit/en/stable/users/molecule_cookbook.html">Molecule cookbook</a> for all the ways to make a <code>Molecule</code>. The crystallographic MCL-1 ligand from PDB ID 6o6f is provided at <code>../structures/606f_ligand.sdf</code>. Note that you don't need to use <code>get_data_file_path</code> as we already know the path.
</div>



```python
# Inspect the Molecule docstring
mcl1_mol = Molecule("../structures/6o6f_ligand.sdf")
top = mcl1_mol.to_topology()
```

We will cover creating a topology for a protein-ligand complex this afternoon.

<a id="interchange"></a>
## 3. `Interchange` objects contain fully parameterised systems with all the information needed to start a simulation

Now we've specified our force field and our chemical system using classes from the OpenFF Tools package (`ForceField`, `Molecule`, and `Topology`), and we want to apply our force field to our chemical topologies (parameterisation).

To do this, we'll use the `Interchange` class from the OpenFF Interchange package, which stores a fully-parameterised molecular system and provides methods to write out simulation-ready input files for a number of software packages. They key objective of Interchange is to provide an intermediate inspectable state after parameterisation and before conversion to an engine-specific format. For most users, an `Interchange` forms the bridge between the OpenFF ecosystem and their simulation software of choice. The current focus is applying SMIRNOFF force fields to chemical topologies and exporting the result to engines preferred by our users. In order of stability, OpenMM, GROMACS, Amber, and LAMMPS are supported. Future development may include support for CHARMM and other engines.

Below is a summary of how data flows through a workflow utilising OpenFF tools, including where Interchange sits in the flow.

<img src="../images/openff_flowchart.png" alt="Description of image" style="max-width: 1000px; display: block; margin-left: auto; margin-right: auto;" />

First, let's recreate our `molecule` and `topology` in case you overwrote them during the previous exercises:


```python
molecule = Molecule.from_smiles("C/C=C/C(=O)[O-]")
topology = molecule.to_topology()
```

An `Interchange` is most commonly constructed via the `Interchange.from_smirnoff()` class method. This method takes a SMIRNOFF force field and applies it to a molecular topology. 




```python
from openff.interchange import Interchange

Interchange.from_smirnoff?
```


```python
interchange = Interchange.from_smirnoff(
    force_field=sage,
    topology=topology,
)
interchange
```




    Interchange with 7 collections, non-periodic topology with 11 atoms.



<div class="alert alert-info" style="max-width: 700px; margin-left: auto; margin-right: auto;">
    ℹ️ <code>ForceField.create_interchange(topology)</code> and <code>Interchange.from_smirnoff(force_field, topology)</code> do the same thing - one just wraps the other. You can use whichever, and interpret them as substitutes of one another.
</div>

An `Interchange` object stores all information known about a system; this includes its chemistry, how that chemistry is represented by a force field, and how the system is organized in 3D space. It has five components:

1. **Topology**: Stores chemical information, such as connectivity and formal charges, independently of force field
1. **Collections**: Maps the chemical information to force field parameters. The force field itself is not directly stored
1. **Positions** (optional): Cartesian co-ordinates of atoms
1. **Box vectors** (optional): Periodicity information
1. **Velocities** (optional): Cartesian velocities of atoms

Let's inspect each of these.

The `Interchange.topology` attribute carries an object of the same type provided by the toolkit and therefore provides the same API. (In the future this may change).


```python
(
    interchange.topology.n_atoms,
    interchange.topology.n_bonds,
    interchange.topology.molecule(0).to_smiles(),
)
```




    (11, 10, '[H]/[C]([C](=[O])[O-])=[C](/[H])[C]([H])([H])[H]')



The `Interchange.collections` attribute carries a dictionary mapping handler names to `SMIRNOFFCollection` objects. These carry the physical parameters derived from applying the force field to the topology.


```python
[(key, value) for key, value in interchange.collections.items()]
```




    [('Bonds',
      Handler 'Bonds' with expression 'k/2*(r-length)**2', 10 mapping keys, and 6 potentials),
     ('Constraints',
      Handler 'Constraints' with expression '', 5 mapping keys, and 2 potentials),
     ('Angles',
      Handler 'Angles' with expression 'k/2*(theta-angle)**2', 15 mapping keys, and 5 potentials),
     ('ProperTorsions',
      Handler 'ProperTorsions' with expression 'k*(1+cos(periodicity*theta-phase))', 19 mapping keys, and 7 potentials),
     ('ImproperTorsions',
      Handler 'ImproperTorsions' with expression 'k*(1+cos(periodicity*theta-phase))', 9 mapping keys, and 2 potentials),
     ('vdW',
      Handler 'vdW' with expression '4*epsilon*((sigma/r)**12-(sigma/r)**6)', 11 mapping keys, and 5 potentials),
     ('Electrostatics',
      Handler 'Electrostatics' with expression 'coul', 11 mapping keys, and 11 potentials)]



Note that each `SMIRNOFFCollection` specifies an algebraic expression which is used to compute the potential energy.

Let's quickly visualize this molecule with atom indices -- this is helpful for looking up particular parameters.


```python
from rdkit.Chem import Draw
from openff.toolkit.topology import Molecule
from IPython.display import SVG


def mol_with_atom_index(molecule: Molecule, width: int = 300, height: int = 300) -> str:
    molecule_copy = Molecule(molecule)
    molecule_copy._conformers = None

    rdmol = molecule_copy.to_rdkit()

    # Build labels like "C:0", "C:1", "C:2", ...
    atom_labels = {
        atom.GetIdx(): f"{atom.GetSymbol()}:{atom.GetIdx()}"
        for atom in rdmol.GetAtoms()
    }

    drawer = Draw.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    for idx, label in atom_labels.items():
        opts.atomLabels[idx] = label

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, rdmol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


SVG(mol_with_atom_index(molecule))
```




    
![svg](output_40_0.svg)
    



The `key_map` attribute of a `SMIRNOFFCollection` maps a `TopologyKey` (such as a `BondKey`) to a `PotentialKey`, which identifies unique parameters:


```python
collection = interchange.collections["Bonds"]
collection.key_map
```




    {BondKey with atom indices (0, 1): PotentialKey associated with handler 'Bonds' with id '[#6X4:1]-[#6X3:2]',
     BondKey with atom indices (0, 6): PotentialKey associated with handler 'Bonds' with id '[#6X4:1]-[#1:2]',
     BondKey with atom indices (0, 7): PotentialKey associated with handler 'Bonds' with id '[#6X4:1]-[#1:2]',
     BondKey with atom indices (0, 8): PotentialKey associated with handler 'Bonds' with id '[#6X4:1]-[#1:2]',
     BondKey with atom indices (1, 2): PotentialKey associated with handler 'Bonds' with id '[#6X3:1]=[#6X3:2]',
     BondKey with atom indices (1, 9): PotentialKey associated with handler 'Bonds' with id '[#6X3:1]-[#1:2]',
     BondKey with atom indices (2, 3): PotentialKey associated with handler 'Bonds' with id '[#6X3:1]-[#6X3:2]',
     BondKey with atom indices (2, 10): PotentialKey associated with handler 'Bonds' with id '[#6X3:1]-[#1:2]',
     BondKey with atom indices (3, 4): PotentialKey associated with handler 'Bonds' with id '[#6X3:1](~[#8X1])~[#8X1:2]',
     BondKey with atom indices (3, 5): PotentialKey associated with handler 'Bonds' with id '[#6X3:1](~[#8X1])~[#8X1:2]'}



We can see that the C=C bond (indices (1,2)) is associated with a potential key with the SMIRKS pattern `[#6X3:1]=[#6X3:2]` (specifying any two carbons bonded to 3 atoms connected by a double bond). Note that the (1,0) C-C bond is matched by the SMRIKS `[#6X3:1]-[#6X3:2]`, which specifies the atoms in the same way, showing that the parameters have been assigned by directly using information about the bond. This contrasts to traditional atom typing approaches, where information about the bond would be implicitly encoded in the atom types used to assign the parameters. Another example of this "direct chemical perception" is the assignment of the carboxylate carbon-oxygen bond parameters, which only match (triply-connected carbon) - (singly-connnected oxygen) bonds when the carbon is bonded to another singly-connected oxygen.

To see the actual parmeters specified for this bond, we can look up the `Potential` objects using the `PotentialKey`s.


```python
for topology_key, potential_key in collection.key_map.items():
    potential = collection.potentials[potential_key]
    print(f"{topology_key} -> {potential}")
```

    atom_indices=(0, 1) bond_order=None -> parameters={'k': <Quantity(478.593862, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.50990609, 'angstrom')>} map_key=None
    atom_indices=(0, 6) bond_order=None -> parameters={'k': <Quantity(715.716502, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.09397889, 'angstrom')>} map_key=None
    atom_indices=(0, 7) bond_order=None -> parameters={'k': <Quantity(715.716502, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.09397889, 'angstrom')>} map_key=None
    atom_indices=(0, 8) bond_order=None -> parameters={'k': <Quantity(715.716502, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.09397889, 'angstrom')>} map_key=None
    atom_indices=(1, 2) bond_order=None -> parameters={'k': <Quantity(904.136474, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.37332262, 'angstrom')>} map_key=None
    atom_indices=(1, 9) bond_order=None -> parameters={'k': <Quantity(772.640205, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.0859499, 'angstrom')>} map_key=None
    atom_indices=(2, 3) bond_order=None -> parameters={'k': <Quantity(534.900401, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.46781367, 'angstrom')>} map_key=None
    atom_indices=(2, 10) bond_order=None -> parameters={'k': <Quantity(772.640205, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.0859499, 'angstrom')>} map_key=None
    atom_indices=(3, 4) bond_order=None -> parameters={'k': <Quantity(1181.50884, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.26002549, 'angstrom')>} map_key=None
    atom_indices=(3, 5) bond_order=None -> parameters={'k': <Quantity(1181.50884, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.26002549, 'angstrom')>} map_key=None


So our C=C bond (indices (1,2)) has a force constant of 904 kcal mol<sup>-1</sup> Å<sup>-2</sup> and an equilibrium bond length of 1.37 Å. Note that the [`ForceField.label_molecules`](https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.typing.engines.smirnoff.ForceField.html#openff.toolkit.typing.engines.smirnoff.ForceField.label_molecules) method is also useful for checking which parameters will be applied to your molecule.

<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Have a look at the "Angles", "ProperTorsions", and "ImproperTorsions" applied. Where are the "ImproperTorsions" applied and why?
</div>


```python
# For example, the ImproperTorsions collection:
collection = interchange.collections["ImproperTorsions"]
collection.key_map

# These enforce planarity around the double bond and carboxylate group
```




    {ImproperTorsionKey with atom indices (1, 0, 2, 9), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (1, 2, 9, 0), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (1, 9, 0, 2), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (2, 1, 3, 10), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (2, 3, 10, 1), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (2, 10, 1, 3), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[*:3])~[*:4]', mult 0,
     ImproperTorsionKey with atom indices (3, 2, 4, 5), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]', mult 0,
     ImproperTorsionKey with atom indices (3, 4, 5, 2), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]', mult 0,
     ImproperTorsionKey with atom indices (3, 5, 2, 4), mult 0: PotentialKey associated with handler 'ImproperTorsions' with id '[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]', mult 0}



<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Change the molecule from the anion to the neutral form by adding a hydrogen to the SMILES. How does this affect the bond strengths of the two carboxyl oxygens?
</div>


```python
bond1_indices, bond2_indices = (3,4), (3,5)
smiles = {"anion": "C/C=C/C(=O)[O-]", "neutral": "C/C=C/C(=O)O"}

for name, smiles in smiles.items():
    print(f"\n{name} ({smiles}):")
    molecule = Molecule.from_smiles(smiles)
    topology = molecule.to_topology()
    interchange = Interchange.from_smirnoff(
        force_field=sage,
        topology=topology,
    )
    collection = interchange.collections["Bonds"]
    bond1 = collection[bond1_indices]
    bond2 = collection[bond2_indices]
    print(f"  Bond {bond1_indices}: {bond1}")
    print(f"  Bond {bond2_indices}: {bond2}")
```

    
    anion (C/C=C/C(=O)[O-]):
      Bond (3, 4): parameters={'k': <Quantity(1181.50884, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.26002549, 'angstrom')>} map_key=None
      Bond (3, 5): parameters={'k': <Quantity(1181.50884, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.26002549, 'angstrom')>} map_key=None
    
    neutral (C/C=C/C(=O)O):


      Bond (3, 4): parameters={'k': <Quantity(1523.99024, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.22481931, 'angstrom')>} map_key=None
      Bond (3, 5): parameters={'k': <Quantity(693.059897, 'kilocalorie_per_mole / angstrom ** 2')>, 'length': <Quantity(1.36024439, 'angstrom')>} map_key=None


Finally, `interchange.box` and `interchange.velocities` are `None`, although `interchange.positions` is populated because we passed a topology with a molecule that had a defined conformer, so `from_smirnoff` set atomic positions from this information:


```python
interchange.positions, interchange.box, interchange.velocities
```




    (None, None, None)



An `Interchange` handles all the information required to run a simulation and allows us to export input files for our engine of choice (OpenMM, GROMACS, LAMMPS, and Amber are all supported)! Let's run a simulation.

Note that since the `Interchange` only contains crontonate and has no box vectors, this would correspond to a vacuum simulation.  We can use [`PACKMOL`](https://m3g.github.io/packmol/) to generate initial positions for a box of water and our solute. Let's use neutral crotonoic acid as our solute so we don't have to worry about neutralising the box.


```python
from openff.interchange.components._packmol import UNIT_CUBE, pack_box
from openff.toolkit import unit

water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
solute = Molecule.from_smiles("C/C=C/C(=O)[OH]") # neutral crotonoic acid

# Naming the residue is not needed to parameterize the system or run the simulation, but makes visualization easier
for atom in water.atoms:
    atom.metadata["residue_name"] = "HOH"

# Generate initial positions using OpenFF's PACKMOL interface. Note that
# using a cubic box is a simple but inefficient choice -- a rhombic
# dodecahedron that provides the same solute separation has only ~ 71 % of
# the volume.
topology = pack_box(
    molecules=[solute, water],
    number_of_copies=[1, 1000],
    box_vectors=3.5 * UNIT_CUBE * unit.nanometer,
)

# Parameterise with Sage, which contains parameters for TIP3P water
interchange = Interchange.from_smirnoff(force_field=sage, topology=topology)
interchange.topology.n_atoms, interchange.box, interchange.positions.shape
```




    (3012,
     <Quantity([[3.5 0.  0. ]
      [0.  3.5 0. ]
      [0.  0.  3.5]], 'nanometer')>,
     (3012, 3))



At this point, we could easily export input files for our simulation engine of choice. For example, for Amber:


```python
interchange.to_amber(prefix="ligand")
```

    /opt/conda/envs/openff-env/lib/python3.12/site-packages/openff/interchange/components/mdconfig.py:502: UserWarning: Ambiguous failure while processing constraints. Constraining h-bonds as a stopgap.
      warnings.warn(
    /opt/conda/envs/openff-env/lib/python3.12/site-packages/openff/interchange/components/mdconfig.py:430: SwitchingFunctionNotImplementedWarning: A switching distance 8.0 angstrom was specified by the force field, but Amber does not implement a switching function. Using a hard cut-off instead. Non-bonded interactions will be affected.
      warnings.warn(



```python
# Check the new files
! ls
```

    complex.gro		 protein_ligand_complex_parameterisation_and_md.ipynb
    complex.top		 small_molecule_parameterisation.ipynb
    complex_pointenergy.mdp  topology.json
    ligand.inpcrd		 trajectory.dcd
    ligand.prmtop		 trajectory_gpu.dcd
    ligand_pointenergy.in


Here, we'll export to OpenMM and run a short simulation directly from the noteboook. We can create an OpenMM `Simulation` object from the `Interchange` and run for a specified wall clock time using `runForClockTime` (the simluation time will depend on how quickly it runs on your machine). We keep the volume ($V$), number of particles ($N$), and average temperature ($T$) (using the LangevinMiddleIntegrator) constant and the simulation corresponds to the $NVT$ ensemble.


```python
import openmm
import openmm.unit
from openff.interchange import Interchange
import mdtraj
import nglview


def run_openmm(
    interchange: Interchange,
    reporter_frequency: int = 50, # Decrease this to save more frames!
    trajectory_name: str = "small_mol_solvated.dcd",
):
    simulation = interchange.to_openmm_simulation(
        integrator=openmm.LangevinMiddleIntegrator(
            300 * openmm.unit.kelvin,
            1 / openmm.unit.picosecond,
            0.002 * openmm.unit.picoseconds,
        ),
    )

    dcd_reporter = openmm.app.DCDReporter(trajectory_name, reporter_frequency)
    simulation.reporters.append(dcd_reporter)

    simulation.context.setVelocitiesToTemperature(300 * openmm.unit.kelvin)
    simulation.runForClockTime(10 * openmm.unit.second)


def visualise_traj(
    topology: Topology, filename: str = "small_mol_solvated.dcd"
) -> nglview.NGLWidget:
    """Visualise a trajectory using nglview."""
    traj = mdtraj.load(
        filename,
        top=mdtraj.Topology.from_openmm(topology.to_openmm()),
    )

    view = nglview.show_mdtraj(traj)
    view.add_representation("licorice", selection="water")

    return view


run_openmm(interchange)
visualise_traj(interchange.topology)
```


    NGLWidget(max_frame=49)


<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> (Only complete this if you have time -- otherwise proceed to section 4.) Create an Interchange object for an MCL-1 ligand. Inspect the parameters assigned and run short simulation as above.
</div>

<a id="gnn_charges"></a>
## 4. Graph Neural Networks Allow Fast Assignment of Partial Charges

You might notice that [Sage](https://github.com/openforcefield/openff-forcefields/blob/main/openforcefields/offxml/openff-2.2.1.offxml) doesn't contain tabulated charges for most atomic environments in the way it does for all other terms in the force field. Instead, it specifies:
```
<ToolkitAM1BCC version="0.3"></ToolkitAM1BCC>
```
which means that partial charges will be calculated using the common AM1-BCC method. Charges from a semi-empirical quantum chemistry calculation (Austin Model 1) are corrected (bond charge correction) to approximate charges obtained by fitting to the electrostatic potential at the HF/6-31G* level (see [Jakalian et al.](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1096-987X(20000130)21:2%3C132::AID-JCC5%3E3.0.CO;2-P)). Unfortunately, AM1-BCC scales 𝒪(N<sup>2</sup>) in the number of atoms N, making it prohibitively slow for large molecules and biopolymers.

Methods which assign partial charges using graph neural networks offer rapid assignment with better scaling. They also offer the possibility of going beyond traditionally affordable QM levels of theory by training to quickly reproduce charges from expensive calculations. For example, [EspalomaCharge](https://pubs.acs.org/doi/full/10.1021/acs.jpca.4c01287) is fit to AM1-BCC charges and offers 𝒪(N<sup>2</sup>) scaling, while [Adams et al.](https://chemrxiv.org/engage/chemrxiv/article-details/6839c94c3ba0887c33d2cd8e) trained models to reproduce atoms-in-molecules charges and electrostatic potentials obtained at a high level of theory. Here, we'll use OpenFF's [AshGC](https://zenodo.org/records/15770227/files/AshGC_methods_2025-06-30.pdf?download=1) model, which is trained to reproduce AM1-BCC charges.

<div class="alert alert-warning" style="max-width: 700px; margin-left: auto; margin-right: auto;">
    ⚠️ OpenFF 2.2.1 has not been explicitly trained and validated with AshGC charges. However, the 2.3.0 release will be, and is expected imminently. AshGC charges will be used as default and will be specified in the <code>.offxml</code> file, so there will be no need to call <code>Molecule.assign_partial_charges</code> as shown below.
</div>


```python
from openff.toolkit import Molecule, ForceField
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

# Disable RDKit warnings to avoid misleading NAGL warnings
# (see https://github.com/openforcefield/openff-nagl/issues/198)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   

# OpenFF NAGL store models as PyTorch files.
ASH_GC_MODEL = "openff-gnn-am1bcc-0.1.0-rc.3.pt"
molecule = Molecule("../structures/6o6f_ligand.sdf")
```


```python
molecule.assign_partial_charges?
```

First, let's assign charges the traditional way with AM1-BCC and check how long this takes...


```python
%%time
molecule_am1bcc = Molecule(molecule)
molecule_am1bcc.assign_partial_charges(
    partial_charge_method="am1bcc",
)
```

    CPU times: user 64.1 ms, sys: 7.24 ms, total: 71.3 ms
    Wall time: 22.6 s


Now, let's try AshGC


```python
%%time
molecule_ashgc = Molecule(molecule)
molecule_ashgc.assign_partial_charges(
    partial_charge_method=ASH_GC_MODEL,
    toolkit_registry=NAGLToolkitWrapper(),
)
```

    CPU times: user 1.13 s, sys: 32.9 ms, total: 1.17 s
    Wall time: 1.1 s


Finally, let's create an `Interchange` with our AshGC charges, making sure to specify `charge_from_molecules` so that we don't replace them with `AM1BCC` charges:


```python
# normally when we call `ForceField.create_interchange` or `ForceField.create_openmm_system`, the toolkit will call
# AMBERTools or OEChem to assign partial charges, since that's what's in the force field file. A future OpenFF release
# which uses NAGL for charge assignment will encode this instruction in the force field file itself, but until that we
# can use the `charge_from_molecules` argument to tell it to use the charges that we just assigned# for more, see:
# https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.typing.engines.smirnoff.ForceField.html#openff.toolkit.typing.engines.smirnoff.ForceField.create_openmm_system
interchange = sage.create_interchange(
    molecule_ashgc.to_topology(),
    charge_from_molecules=[molecule_ashgc],
)
```

<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Exercise:</b> Compare the charges obtained with AM1-BCC and AshGC by looking at the <code>Molecule.partial_charges</code> attribute. How big are these differences on average? What is the largest difference? Which atom are these on? The <code>np.max</code> function may be useful.
</div>


```python
# Compare charges assigned with AM1-BCC and AshGC...
import numpy as np
print(f"AM1 BCC charges: {molecule_am1bcc.partial_charges}")
print(f"AshGC charges:   {molecule_ashgc.partial_charges}")

differences = molecule_am1bcc.partial_charges - molecule_ashgc.partial_charges
differences_by_atom_index = {idx: diff.magnitude for idx, diff in enumerate(differences)}
print(f"Differences by atom index: {differences_by_atom_index}")

max_difference = np.max(np.abs(differences.magnitude))
print(f"Max difference: {max_difference} e")

mean_difference = np.mean(np.abs(differences.magnitude))
print(f"Mean absolute difference: {mean_difference} e")

# Get the atom index with the largest difference
atom_index = np.argmax(np.abs(differences.magnitude))
print(f"Largest absolute difference is for atom index {atom_index}, which is a {molecule_ashgc.atoms[atom_index].symbol} atom")
```

    AM1 BCC charges: [-0.12740000000000004 -0.8283 -0.8283 -0.33690000000000003 -0.633 0.14839999999999998 0.19979999999999998 0.20379999999999998 -0.09940000000000004 -0.13500000000000004 -0.15200000000000002 -0.09040000000000004 -0.09040000000000004 -0.11100000000000004 -0.07100000000000004 -0.03410000000000003 -0.12800000000000003 -0.13800000000000004 -0.07840000000000004 0.9082 0.012399999999999965 0.04109999999999996 0.059599999999999966 -0.11570000000000004 -0.11560000000000004 -0.09030000000000005 -0.053300000000000035 -0.03870000000000003 0.04569999999999996 0.04569999999999996 0.056699999999999966 0.056699999999999966 0.03769999999999996 0.03769999999999996 0.039199999999999964 0.039199999999999964 0.14199999999999996 0.12199999999999996 0.049699999999999966 0.049699999999999966 0.049699999999999966 0.049699999999999966 0.15599999999999997 0.18099999999999997 0.047699999999999965 0.047699999999999965 0.13699999999999998 0.16099999999999998 0.04469999999999996 0.04469999999999996 0.08069999999999995] elementary_charge
    AshGC charges:   [-0.10291757708524957 -0.8212745068046976 -0.8212745068046976 -0.3278780457947184 -0.656814920661204 0.13626032793784842 0.22055995190406547 0.2206000658299993 -0.09947414970134988 -0.14171894168590798 -0.17288624024128213 -0.09598572826122537 -0.09598569845890298 -0.0818095085594584 -0.1024471399757792 -0.04326643323635354 -0.12058025872444406 -0.08199206268524423 -0.07742916321491494 0.9052312495734762 0.027429900186903337 0.0331591638352941 0.0954736592795919 -0.10021315043901696 -0.1429999051067759 -0.07631889259552255 -0.09621359681820169 -0.03022590180968537 0.044725027921445226 0.044725027921445226 0.03946309262777076 0.03946309262777076 0.03401340270305381 0.03401340270305381 0.04491722309852347 0.04491722309852347 0.14514142000938163 0.11216316843295798 0.047582616153008794 0.047582616153008794 0.047582616153008794 0.047582616153008794 0.15584553504253135 0.13253731751704917 0.056406603249556875 0.056406603249556875 0.14515182101989493 0.1600255640771459 0.05733717704082236 0.05733717704082236 0.05607166612411246] elementary_charge
    Differences by atom index: {0: np.float64(-0.024482422914750473), 1: np.float64(-0.007025493195302435), 2: np.float64(-0.007025493195302435), 3: np.float64(-0.009021954205281624), 4: np.float64(0.023814920661203942), 5: np.float64(0.012139672062151552), 6: np.float64(-0.020759951904065488), 7: np.float64(-0.016800065829999322), 8: np.float64(7.41497013498349e-05), 9: np.float64(0.006718941685907948), 10: np.float64(0.020886240241282106), 11: np.float64(0.005585728261225331), 12: np.float64(0.0055856984589029435), 13: np.float64(-0.029190491440541644), 14: np.float64(0.031447139975779165), 15: np.float64(0.009166433236353508), 16: np.float64(-0.007419741275555974), 17: np.float64(-0.05600793731475581), 18: np.float64(-0.0009708367850850969), 19: np.float64(0.0029687504265237807), 20: np.float64(-0.015029900186903372), 21: np.float64(0.00794083616470586), 22: np.float64(-0.03587365927959193), 23: np.float64(-0.015486849560983076), 24: np.float64(0.027399905106775868), 25: np.float64(-0.0139811074044775), 26: np.float64(0.04291359681820165), 27: np.float64(-0.008474098190314663), 28: np.float64(0.0009749720785547367), 29: np.float64(0.0009749720785547367), 30: np.float64(0.01723690737222921), 31: np.float64(0.01723690737222921), 32: np.float64(0.003686597296946155), 33: np.float64(0.003686597296946155), 34: np.float64(-0.005717223098523509), 35: np.float64(-0.005717223098523509), 36: np.float64(-0.003141420009381668), 37: np.float64(0.009836831567041973), 38: np.float64(0.002117383846991172), 39: np.float64(0.002117383846991172), 40: np.float64(0.002117383846991172), 41: np.float64(0.002117383846991172), 42: np.float64(0.0001544649574686252), 43: np.float64(0.0484626824829508), 44: np.float64(-0.00870660324955691), 45: np.float64(-0.00870660324955691), 46: np.float64(-0.00815182101989495), 47: np.float64(0.0009744359228540667), 48: np.float64(-0.0126371770408224), 49: np.float64(-0.0126371770408224), 50: np.float64(0.02462833387588749)}
    Max difference: 0.05600793731475581 e
    Mean absolute difference: 0.013057460803529108 e
    Largest absolute difference is for atom index 17, which is a C atom


<a id="summary"></a>
## 5. Conclusions

* The `ForceField` class from the OpenFF Toolkit allows force fields to be easily loaded and inspected.
* The `Molecule` and `Topology` classes from the OpenFF Toolkit allow us to represent a chemical system, independently from the force field.
* The `Interchange` class from OpenFF Interchange handles fully parameterised systems with all the information required to start a simulation. Exporting to the simluation engine of your choice is simple, and we can easily run a simulation with OpenMM without leaving the notebook!
* Graph neural networks can provide fast and high-quality conformer-independent partial charges.

<a id="further_materials"></a>
## 6. There's Lots More to OpenFF!

A variety of example notebooks for OpenFF software are provided [here](https://docs.openforcefield.org/en/latest/examples.html). A few which are particularly relevant are:

- [Compute conformer energies for a small molecule](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/conformer_energies/conformer_energies.html)
- [Modifying a SMIRNOFF force field](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/forcefield_modification/forcefield_modification.html)
- [Inspect parameters assigned to specific molecules](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/inspect_assigned_parameters/inspect_assigned_parameters.html)

<div class="alert alert-success" style="max-width: 500px; margin-left: auto; margin-right: auto; border-left: 6px solid #5cb85c; background-color: #f1fff1;">
    ✏️ <b>Extra Exercises:</b> Based on the above tutorials, can you:
            <ul>
            <li>Generate several conformers for one of your MCL-1 ligands and compute their relative energies using OpenFF 2.2.1?</li>
            <li>Modify OpenFF 2.2.1 to change some of the parameters applied to one of your MCL-1 ligands? Minimise the ligand with this new force field and see how your changes influence the conformation.</li>
            <li>Analyse which parameters are shared and which are only applied to one or few molecules for a set of MCL-1 ligands?</li>
            </ul>
</div>
