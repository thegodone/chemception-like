import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools

frame = PandasTools.LoadSDF('dhfr_3d.sd',smilesName='smiles', molColName='Molecule', includeFingerprints=False)

frame.to_hdf('Sutherland_DHFR.h5', key='data', mode='w')