from seaborn import color_palette
from numpy import logspace

PM2ANGSTROM = 0.01
NM2ANGSTROM = 10
ANGSTROM2NM = 0.1
PM2NM = 0.001

DIM_LANDSCAPE = (7.25, 4.45)
DIM_PORTRAIT = (3.45, 4.45)
DEFAULT_DPI = 600
DEFAULT_DPI_MULTI = 300
COLORS = color_palette("hls", 6)
DEFAULT_SEED = 1
UNIQUE_TOL = 5

PATH_FIG_COM = "figures/center_of_mass/"
PATH_FIG_GRID = "figures/grids/"
PATH_FIG_LENG = "figures/leng_structures/"
PATH_FIG_TRAJ = "figures/trajectories/"
PATH_FIG_CON = "figures/convergence/"
PATH_FIG_EDT = "figures/energy_during_trajectory/"
PATH_FIG_CME = "figures/cumulative_min_energy/"
PATH_FIG_PRES = "figures/presentation/"
PATH_FIG_TEST = "figures/tests/"
PATH_FIG_TIME = "figures/timing/"

PATH_ANI_COM = "animations/center_of_mass/"
PATH_ANI_ROT = "animations/rotations/"
PATH_ANI_GRID = "animations/grids/"
PATH_ANI_TRAJ = "animations/trajectories/"
PATH_ANI_LENG = "animations/leng_structures/"

PATH_GENERATED_GRO_FILES = "../../nobackup/data/generated_gro_files/"
PATH_BASE_GRO_FILES = "provided_data/base_gro_files/"
PATH_COMPARE_GRIDS = "../../nobackup/data/compare_grids/"
PATH_GRO_RESULTS = "../../nobackup/data/gro_results/"
PATH_OMM_RESULTS = "../../nobackup/data/openMM_results/"
PATH_SAVED_GRIDS = "../../nobackup/data/saved_grids/"
PATH_TOPOL = "provided_data/topologies/"
PATH_FF = "provided_data/force_fields/"
PATH_TIME_SIM = "../../nobackup/data/time_simulations/"
PATH_VIOLIN_DATA = "../../nobackup/data/violin_data/"

PATH_GROMACS = "../../nobackup/gromacs/"
PATH_SCRIPTS = "../../nobackup/gromacs/SCRIPTS/"

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
#DEFAULT_DISTANCES = (0.25, 0.05, 0.05, 0.05, 0.05)
DEFAULT_DISTANCES = (0.26, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005)
#DEFAULT_NS = logspace(0.5, 5, num=30, dtype=int)
DEFAULT_NS = [3,    4,    5,    6,    8,   10,   12,   15,   18,   23,
         30,   34,   41,   50,   62,   75,   100, 112,  137,  167,  204,
        249,  300,  370,  452,  500,  672,  819, 1000]
DEFAULT_NS_TIME = [3,    4,    5,    6,    8,   10,   12,   15,   18,   23,
         30,   34,   41,   50,   62,   75,   100, 112,  137,  167,  204,
        249,  300]
GROMACS_NS = [54, 96, 150, 216, 384, 600, 864, 1350, 1944, 3174, 5400, 6936, 10086, 15000, 23064, 33750, 60000] # 19
MINI_DEFAULT_NS = [30, 100, 300]
