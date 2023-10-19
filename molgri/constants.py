"""
The central source of constants.
"""

from seaborn import color_palette
from pkg_resources import resource_filename
from scipy.constants import pi

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
EXTENSION_FIGURES = "pdf"
EXTENSION_ANIMATIONS = "gif"
EXTENSION_TRAJECTORY = "xtc"
EXTENSION_TOPOLOGY = "gro"
EXTENSION_LOGGING = "log"

# here write non-user-defined paths
PATH_USER_PATHS = resource_filename("molgri", "paths.py")
PATH_EXAMPLES = resource_filename("molgri", "examples/")

# algorithms
DEFAULT_ALGORITHM_O = "ico"
DEFAULT_ALGORITHM_B = "cube4D"
GRID_ALGORITHMS_3D = ("randomS", "cube3D", "ico")
GRID_ALGORITHMS_4D = ("randomQ", "cube4D", "fulldiv")
ALL_GRID_ALGORITHMS = GRID_ALGORITHMS_3D + GRID_ALGORITHMS_4D
assert DEFAULT_ALGORITHM_O in GRID_ALGORITHMS_3D
assert DEFAULT_ALGORITHM_B in GRID_ALGORITHMS_4D
FULL_GRID_ALG_NAMES = ("random points on a sphere", "3D cube grid", "icosahedron grid", "random quaternions",
                          "4D cube grid", "full division of Cube4D")
SHORT_GRID_ALG_NAMES = ("random S.", "cube 3D", "icosahedron", "random q.", "cube 4D", "full div.")
assert len(ALL_GRID_ALGORITHMS) == len(FULL_GRID_ALG_NAMES) == len(SHORT_GRID_ALG_NAMES)
NAME2PRETTY_NAME = {n: pn for n, pn in zip(ALL_GRID_ALGORITHMS, FULL_GRID_ALG_NAMES)}
NAME2SHORT_NAME = {n: pn for n, pn in zip(ALL_GRID_ALGORITHMS, SHORT_GRID_ALG_NAMES)}

# energies
ENERGY_TYPES = ("LJ Energy [kJ/mol]", "Dis.corr. [kJ/mol]", "Coulomb [kJ/mol]",
                "Potential [kJ/mol]")
ENERGY_NO_UNIT = ("LJ (SR)", "Disper. corr.", "Coulomb (SR)", "Potential")
ENERGY_SHORT_TYPES = ("LJ", "Dis", "Coulumb", "pot")
assert len(ENERGY_TYPES) == len(ENERGY_SHORT_TYPES)
ENERGY_SHORT2FULL = {n: pn for n, pn in zip(ENERGY_SHORT_TYPES, ENERGY_TYPES)}
ENERGY_FULL2SHORT = {n: pn for n, pn in zip(ENERGY_TYPES, ENERGY_SHORT_TYPES)}
ENERGY2SHORT = {n: pn for n, pn in zip(ENERGY_NO_UNIT, ENERGY_SHORT_TYPES)}

MINI_NS = (20, 50, 80)

SMALL_NS = (10,   12,   15,   18,   23, 30,   34,   41,   50,   62,   75,   100, 112,  137,  167,  204,
            249,  300)

DEFAULT_NS = (10,   12,   15,   18,   23,
              30,   34,   41,   50,   62,   75,   100, 112,  137,  167,  204,
              249,  300,  370,  452,  500,  672,  819, 1000)

# statistics
DEFAULT_ALPHAS_4D = (pi / 6, 2 * pi / 6, 3 * pi / 6, 4 * pi / 6, 5 * pi / 6)
TEXT_ALPHAS_4D = [r'$\frac{\pi}{6}$', r'$\frac{2\pi}{6}$', r'$\frac{3\pi}{6}$', r'$\frac{4\pi}{6}$',
                 r'$\frac{5\pi}{6}$']

DEFAULT_ALPHAS_3D = (pi/12, pi / 6, 3*pi/12, 2 * pi / 6, 5 * pi / 12)
TEXT_ALPHAS_3D = [r'$\frac{\pi}{12}$', r'$\frac{\pi}{6}$', r'$\frac{3\pi}{12}$', r'$\frac{2\pi}{6}$',
                  r'$\frac{5\pi}{12}$']
# DEFAULT_ALPHAS_4D = DEFAULT_ALPHAS_3D
# TEXT_ALPHAS_4D = TEXT_ALPHAS_3D

EXTENSIONS_STR = ('PSF', 'TOP', 'PRMTOP', 'PARM7', 'PDB', 'ENT', 'XPDB', 'PQR', 'GRO', 'CRD', 'PDBQT', 'DMS',
              'TPR', 'MOL2', 'DATA', 'LAMMPSDUMP', 'XYZ', 'TXYZ', 'ARC', 'GMS', 'CONFIG', 'HISTORY', 'XML',
              'MMTF', 'GSD', 'MINIMAL', 'ITP', 'IN', 'FHIAIMS', 'PARMED', 'RDKIT', 'OPENMMTOPOLOGY',
              'OPENMMAPP')
EXTENSIONS_TRJ = ("XTC", "TRR", "CHEMFILES", "DCD", "DATA", "TNG", "XYZ", "TXYZ", "ARC", "GSD", "GMS", "PDB", "ENT",
                  "PDBQT")

# you may add at the end but don't change the order
CELLS_DF_COLUMNS = ["N points", "Radius [A]", "Voranoi area [A^2]", "Ideal area [A^2]",
                    "Grid creation time [s]", "Grid tesselation time [s]"]
