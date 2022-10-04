from objects_3d.abstractshape import AbstractShape
from objects_3d.molecule import Atom, Molecule, H2O, HF
from scipy.constants import pi
from my_constants import *
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.axis import Axis
from collections import defaultdict
from typing import TextIO


class ShapeSet(object):

    def __init__(self, all_objects: list or AbstractShape, num_dim: int = 3, name: str = "ShapeSet",
                 time_step: int = 0):
        """
        An object joining several AbstractShape objects so they can be translated, rotated or drawn together.
        Using 'which' command, only a part of the collection can be manipulated.
        """
        self.num_dim = num_dim
        self.time_step = time_step
        self.name = name
        if type(all_objects) == AbstractShape:
            self.all_objects = [all_objects]
        else:
            self.all_objects = all_objects
        assert [type(x) == AbstractShape for x in self.all_objects], "All simulated objects must be type(AbstractShape)"
        assert [x.dimension == self.num_dim for x in self.all_objects], "All objects must have the same" \
                                                                        "number of dimensions as the set"

    def draw_objects(self, axis: Axis, which: list = None, **kwargs) -> list:
        """
        The only method that actually draws. All other methods should call this method to draw.

        Args:
            axis: axis on which to draw
            which: list of 1s and 0s as long as self.all_objects or None - which objects should be drawn. If None, all.
            **kwargs: forwarded to AbstractShape.draw(), e.g. show_basis = True
        """
        all_images = []
        which = self._select_which_objects(which)
        for i, obj in enumerate(self.all_objects):
            if which[i]:
                all_images.extend(obj.draw(axis, **kwargs))
        return all_images

    def _select_which_objects(self, which: list or None):
        """
        Helper function to create a filter which objects to use from self.all_objects. If which=None, use all.
        """
        if which is None:
            which = [1] * len(self.all_objects)
        else:
            assert len(which) == len(self.all_objects), "which needs to provide 1/0 for each object in self.all_objects"
            assert all([x in [1, 0] for x in which]), "which cannot have values other than 1 or 0"
        return which

    def _manipulate_once(self, which_action: str, vector: ArrayLike, which: list = None, method: str = "euler_123",
                         inverse: bool = False):
        """
        Selects appropriate which_action and forwards to the corresponding function.

        Args:
            which_action: options: ['translate', 'rotate_objects_about_origin', 'rotate_objects_about_body']
            vector: vector for translation or vector/float of rotational angles
            which: list of 1s and 0s as long as self.all_objects or None - which objects should be drawn. If None, all
            method: what method of rotation
        """
        for j, _ in enumerate(self.all_objects):
            if which[j]:
                if which_action == "translate":
                    self.all_objects[j].translate(vector)
                elif which_action == "rotate_about_body":
                    return self.all_objects[j].rotate_about_body(vector, method=method, inverse=inverse)
                elif which_action == "rotate_about_origin":
                    return self.all_objects[j].rotate_about_origin(vector, method=method, inverse=inverse)
                else:
                    raise NotImplementedError("Only actions translate, rotate_about_body and" +
                                              "rotate_abut_origin are implemented.")

    def _manipulate_objects(self, which_action: str, vector: ArrayLike, which: list = None,
                            num_steps: int = 1, method: str = "euler_123", inverse=False, **kwargs):
        """
        Helper method that implements the background of translate_objects, rotate_objects_about_origin and
        rotate_objects_about_body.

        Args:
            num_steps: in how many steps the movement should happen
            the rest of arguments forwarded to self._manipulate_once()
        """
        vector = np.array(vector)
        which = self._select_which_objects(which)
        # can be done in several steps for a better illustration in simulation
        for i in range(num_steps):
            # do it for all objects selected with which command
            self._manipulate_once(which_action=which_action, vector=vector/num_steps, which=which, method=method,
                                  inverse=inverse)
            self.time_step += 1

    def translate_objects(self, vector: ArrayLike, which: list = None, **kwargs):
        """
        Translate all or some of objects for vector. Can be drawn or not, in multiple steps or not. See
        self._manipulate_objects for details.
        """
        self._manipulate_objects("translate", vector, which=which, **kwargs)

    def rotate_objects_about_origin(self, vector: ArrayLike, which: list = None, **kwargs):
        """
        Rotate all or some of objects about origin for a set of angles. Can be drawn or not, in multiple steps or not.
        See self._manipulate_objects for details.
        """
        self._manipulate_objects("rotate_about_origin", vector, which=which, **kwargs)

    def rotate_objects_about_body(self, vector: ArrayLike, which: list = None, **kwargs):
        """
        Rotate all or some of objects about body for a set of angles. Can be drawn or not, in multiple steps or not.
        See self._manipulate_objects for details.
        """
        self._manipulate_objects("rotate_about_body", vector, which=which, **kwargs)


class AtomSet(ShapeSet):

    def __init__(self, all_objects: list or AbstractShape, file_gro: TextIO = None,
                 file_xyz: TextIO = None, **kwargs):
        super().__init__(all_objects, **kwargs)
        assert [type(x) == Atom for x in self.all_objects], "All simulated objects must be Atoms"
        self.belongings = self._group_by_belonging()
        self.file_gro = file_gro
        self.file_xyz = file_xyz
        self.num_atoms = len(self.all_objects)

    def _group_by_belonging(self) -> dict:
        try:
            belonging_dic = defaultdict(list)
            for one_object in self.all_objects:
                belonging_dic[one_object.belongs_to].append(one_object)
            belonging_dic = dict(belonging_dic)
            return belonging_dic
        # this happens when initializing MoleculeSet
        except AttributeError:
            return dict()

    def _select_which_objects(self, which: list or None):
        """
        Helper function to create a filter which objects to use from self.all_objects. If which=None, use all.
        """
        if which is None:
            which = [1] * len(self.all_objects)
        elif which in list(self.belongings.keys()):
            return [x.belongs_to == which for x in self.all_objects]
        else:
            assert len(which) == len(self.all_objects), "which needs to provide 1/0 for each object in self.all_objects"
            assert all([x in [1, 0] for x in which]), "which cannot have values other than 1 or 0"
        return which

    # def save_to_xyz(self):
    #     """
    #     If one of the attributes is file_xyz, current coordinates can be converted to a .xyz file format and
    #     written to the corresponding file.
    #     """
    #     if not self.file_xyz:
    #         raise AttributeError("Must set the file_xyz attribute (file to which to save!)")
    #     self.file_xyz.write(str(self.num_atoms) + "\n\n")
    #     for molecule in self.all_objects:
    #         for el, atom in zip(molecule.elements, molecule.atoms):
    #             self.file_xyz.write(f"{el.symbol}\t{atom.position[0]}\t{atom.position[1]}\t{atom.position[2]}\n")
    #     self.file_xyz.write("\n")
    #
    # def save_to_gro(self, residue: str = "SOL", box: ArrayLike = (3, 3, 3)):
    #     """
    #     If one of the attributes is file_gro, current coordinates can be converted to a .gro file format and
    #     written to the corresponding file.
    #
    #     Args:
    #         residue: how to name the residue when saving file, default 'SOL' for solvent
    #         box: size of the boy to write in the file
    #     """
    #     if not self.file_gro:
    #         raise AttributeError("Must set the file_gro attribute (file to which to save!)")
    #     self.file_gro.write(f"{self.name} t={self.time_step}.0\n")
    #     self.file_gro.write(f"{self.num_atoms:5}\n")
    #     self._save_atom_lines_gro(residue=residue)
    #     for box_el in box:
    #         self.file_gro.write(f"\t{box_el}")
    #     self.file_gro.write("\n")

    def _save_atom_lines_gro(self, residue: str = "SOL", atom_num=1, residue_num=1):
        num_atom = atom_num
        num_molecule = residue_num
        for molecule_name, molecule_atoms in self.belongings.items():
            hydrogen_counter = 1
            for atom in molecule_atoms:
                pos_nm = atom.position
                if atom.element.symbol == "O":
                    name = "OW"
                elif atom.element.symbol == "H":
                    name = "HW" + str(hydrogen_counter)
                    hydrogen_counter += 1
                else:
                    name = atom.element.symbol
                self.file_gro.write(f"{num_molecule:5}{residue:5}{name:>5}{num_atom:5}{pos_nm[0]:8.3f}{pos_nm[1]:8.3f}"
                                    f"{pos_nm[2]:8.3f}{0:8.4f}{0:8.4f}{0:8.4f}\n")
                num_atom += 1
            num_molecule += 1

    def _manipulate_objects(self, *args, save_to_gro=False, ** kwargs):
        super()._manipulate_objects(*args, **kwargs)
        if save_to_gro:
            self.save_to_gro()


class MoleculeSet(AtomSet):

    def __init__(self, all_objects: list or AbstractShape, file_gro: TextIO = None, file_xyz: TextIO = None, **kwargs):
        super().__init__(all_objects, file_gro=file_gro, file_xyz=file_xyz, **kwargs)
        assert [type(x) == Molecule for x in self.all_objects], "All objects must be Molecules"
        # changing from AtomSet
        self.belongings = {f"molecule_{i}": key.atoms for i, key in enumerate(self.all_objects)}
        # changing from AtomSet
        self.num_atoms = sum([len(self.all_objects[i].atoms) for i in range(len(self.all_objects))])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    from plotting.plotting_helper_functions import set_axes_equal
    import os
    filename = "../data/generated_gro_files/example_water_set.gro"
    if os.path.exists(filename):
        os.remove(filename)
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    with open(filename, "a") as f:
        water_set = MoleculeSet([HF(), H2O()], file_gro=f)
        water_set.translate_objects([0, 0, 0.35], which=[0, 1])
        water_set.rotate_objects_about_origin(np.array([pi/3, pi/2, pi]), which=[0, 1])
        water_set.draw_objects(ax)
    set_axes_equal(ax)
    plt.show()
    plt.style.use('default')
