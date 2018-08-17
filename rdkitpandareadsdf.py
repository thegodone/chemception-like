import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools

frame = PandasTools.LoadSDF('data/dhfr_3d.sd',smilesName='smiles', molColName='Molecule', includeFingerprints=False)

frame.to_hdf('data/Sutherland_DHFR.h5', key='data', mode='w')
