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

ENDING_GRID_FILES = "npy"
ENDING_FIGURES = "png"

# here write non-user-defined paths
PATH_USER_PATHS = resource_filename("molgri", "paths.py")
PATH_EXAMPLES = resource_filename("molgri", "examples/")

SIX_METHOD_NAMES = ("systemE", "randomE", "randomQ", "cube4D", "cube3D", "ico")
PRETTY_METHOD_NAMES = ("systematic Euler angles", "random Euler angles", "random quaternions", "4D cube grid",
                       "3D cube grid", "icosahedron grid")
NORMAL_SIZE = 500
SMALL_SIZE = 50
SHORT_NAMES = ("system. E.", "random E.", "random q.", "cube 4D", "cube 3D", "icosahedron")
SIZE2NUMBERS = {"normal": NORMAL_SIZE, "small": SMALL_SIZE}
NAME2PRETTY_NAME = {n: pn for n, pn in zip(SIX_METHOD_NAMES, PRETTY_METHOD_NAMES)}
NAME2SHORT_NAME = {n: pn for n, pn in zip(SIX_METHOD_NAMES, SHORT_NAMES)}
NAME2PRETTY_NAME[None] = "optimization run"
FULL_RUN_NAME = "run"
ADDITIONAL_NAMES = ("cube3D_NO", "ico_NO")
MOLECULE_NAMES = ("H2O", "HF", "protein0", "protein1", "protein2", "protein3", "protein4", "protein5", "CL", "NA")
ENERGY_TYPES = ["LJ Energy [kJ/mol]", "Dis.corr. [kJ/mol]", "Coulomb [kJ/mol]",
                "Potential [kJ/mol]"]
ENERGY_SHORT_TYPES = ["LJ", "Dis", "Coulumb", "pot"]
ENERGY_SHORT2FULL = {n: pn for n, pn in zip(ENERGY_SHORT_TYPES, ENERGY_TYPES)}
ENERGY_FULL2SHORT = {n: pn for n, pn in zip(ENERGY_TYPES, ENERGY_SHORT_TYPES)}
DEFAULT_DISTANCES_PROTEIN = (0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25)
DEFAULT_DISTANCES = (0.26, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005)
DEFAULT_NS = [3,    4,    5,    6,    8,   10,   12,   15,   18,   23,
         30,   34,   41,   50,   62,   75,   100, 112,  137,  167,  204,
        249,  300,  370,  452,  500,  672,  819, 1000]
DEFAULT_NS_TIME = [3,    4,    5,    6,    8,   10,   12,   15,   18,   23,
         30,   34,   41,   50,   62,   75,   100, 112,  137,  167,  204,
        249,  300]
GROMACS_NS = [54, 96, 150, 216, 384, 600, 864, 1350, 1944, 3174, 5400, 6936, 10086, 15000, 23064, 33750, 60000] # 19
MINI_DEFAULT_NS = [30, 100, 300]
