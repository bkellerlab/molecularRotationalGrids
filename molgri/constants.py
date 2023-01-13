from seaborn import color_palette
from pkg_resources import resource_filename

PM2ANGSTROM = 0.01
NM2ANGSTROM = 10
ANGSTROM2NM = 0.1
PM2NM = 0.001

DIM_LANDSCAPE = (7.25, 4.45)
DIM_PORTRAIT = (3.45, 4.45)
DIM_SQUARE = (4.45, 4.45)
DEFAULT_DPI = 600
DEFAULT_DPI_MULTI = 300
COLORS = color_palette("hls", 6)
DEFAULT_SEED = 1
UNIQUE_TOL = 5

EXTENSION_GRID_FILES = "npy"
EXTENSION_FIGURES = "png"
EXTENSION_TRAJECTORY = "xtc"
EXTENSION_TOPOLOGY = "gro"

# here write non-user-defined paths
PATH_USER_PATHS = resource_filename("molgri", "paths.py")
PATH_EXAMPLES = resource_filename("molgri", "examples/")

# algorithms
DEFAULT_ALGORITHM_O = "ico"
DEFAULT_ALGORITHM_B = "cube4D"
ZERO_ALGORITHM = "zero"
GRID_ALGORITHMS = ("systemE", "randomE", "randomQ", "cube4D", "cube3D", "ico", "zero")
assert DEFAULT_ALGORITHM_O in GRID_ALGORITHMS and ZERO_ALGORITHM in GRID_ALGORITHMS
FULL_GRID_ALG_NAMES = ("systematic Euler angles", "random Euler angles", "random quaternions", "4D cube grid",
                       "3D cube grid", "icosahedron grid", "no rotation")
SHORT_GRID_ALG_NAMES = ("system. E.", "random E.", "random q.", "cube 4D", "cube 3D", "icosahedron", "no rotat.")
assert len(GRID_ALGORITHMS) == len(FULL_GRID_ALG_NAMES) == len(SHORT_GRID_ALG_NAMES)
NAME2PRETTY_NAME = {n: pn for n, pn in zip(GRID_ALGORITHMS, FULL_GRID_ALG_NAMES)}
NAME2SHORT_NAME = {n: pn for n, pn in zip(GRID_ALGORITHMS, SHORT_GRID_ALG_NAMES)}
ENERGY_TYPES = ("LJ Energy [kJ/mol]", "Dis.corr. [kJ/mol]", "Coulomb [kJ/mol]",
                "Potential [kJ/mol]")
ENERGY_SHORT_TYPES = ("LJ", "Dis", "Coulumb", "pot")
assert len(ENERGY_TYPES) == len(ENERGY_SHORT_TYPES)
ENERGY_SHORT2FULL = {n: pn for n, pn in zip(ENERGY_SHORT_TYPES, ENERGY_TYPES)}
ENERGY_FULL2SHORT = {n: pn for n, pn in zip(ENERGY_TYPES, ENERGY_SHORT_TYPES)}

SMALL_NS = (8,   10,   12,   15,   18,   23, 30,   34,   41,   50,   62,   75,   100, 112,  137,  167,  204,
            249,  300)

DEFAULT_NS = (3,    4,    5,    6,    8,   10,   12,   15,   18,   23,
              30,   34,   41,   50,   62,   75,   100, 112,  137,  167,  204,
              249,  300,  370,  452,  500,  672,  819, 1000)


EXTENSIONS = ('PSF', 'TOP', 'PRMTOP', 'PARM7', 'PDB', 'ENT', 'XPDB', 'PQR', 'GRO', 'CRD', 'PDBQT', 'DMS',
              'TPR', 'MOL2', 'DATA', 'LAMMPSDUMP', 'XYZ', 'TXYZ', 'ARC', 'GMS', 'CONFIG', 'HISTORY', 'XML',
              'MMTF', 'GSD', 'MINIMAL', 'ITP', 'IN', 'FHIAIMS', 'PARMED', 'RDKIT', 'OPENMMTOPOLOGY',
              'OPENMMAPP')

# you may add at the end but don't change the order
CELLS_DF_COLUMNS = ["N points", "Radius [A]", "Voranoi area [A^2]", "Ideal area [A^2]",
                    "Grid creation time [s]", "Grid tesselation time [s]"]
