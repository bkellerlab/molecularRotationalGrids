from molgri.grids.grid import Grid
from molgri.pseudotrajectories.gen_base_gro import TwoMoleculeGro
from molgri.my_constants import *
from molgri.wrappers import time_method


class Pseudotrajectory(TwoMoleculeGro):

    def __init__(self, name_central_gro, name_rotating_gro, grid: Grid, traj_type="full"):
        grid_name = grid.standard_name  # for example ico_500
        pseudo_name = f"{name_central_gro}_{name_rotating_gro}_{grid_name}_{traj_type}"
        self.pt_name = pseudo_name
        super().__init__(name_central_gro, name_rotating_gro, result_name_gro=pseudo_name)
        self.quaternions = grid.as_quaternion()
        self.traj_type = traj_type
        self.name_rotating = name_rotating_gro
        self.decorator_label = f"pseudotrajectory {pseudo_name}"

    def _gen_trajectory(self, frame_index=0) -> int:
        """
        This does not deal with any radii yet, only with rotations.
        Args:
            frame_index:

        Returns:

        """
        frame_index = frame_index
        for one_rotation in self.quaternions:
            initial_atom_set = self.rotating_parser.molecule_set
            initial_atom_set.rotate_about_origin(one_rotation, method="quaternion")
            self._write_current_frame(frame_num=frame_index)
            frame_index += 1
            if self.traj_type == "full":
                for body_rotation in self.quaternions:
                    # rotate there
                    initial_atom_set.rotate_about_body(body_rotation, method="quaternion")
                    self._write_current_frame(frame_num=frame_index)
                    # rotate back
                    initial_atom_set.rotate_about_body(body_rotation, method="quaternion", inverse=True)
                    frame_index += 1
            initial_atom_set.rotate_about_origin(one_rotation, method="quaternion", inverse=True)
        return frame_index

    # TODO: translational grid shouldn't be an input here
    def generate_pseudotrajectory(self, initial_distance_nm=0.26, radii=DEFAULT_DISTANCES) -> int:
        index = 0
        # initial set-up of molecules
        self.rotating_parser.molecule_set.translate([0, 0, initial_distance_nm])
        self._write_current_frame(index)
        index += 1
        if self.traj_type == "circular":
            self._gen_trajectory(frame_index=index)
        elif self.traj_type == "full":
            # go over different radii
            for shell_d in radii[1:]:
                index = self._gen_trajectory(frame_index=index)
                self.rotating_parser.molecule_set.translate([0, 0, shell_d])
        else:
            raise ValueError(f"{self.traj_type} not correct trajectory type, try 'full' or 'circular'.")
        self.f.close()
        return index

    @time_method
    def generate_pt_and_time(self, *args, **kwargs) -> int:
        return self.generate_pseudotrajectory(*args, **kwargs)


if __name__ == "__main__":
    from molgri.grids.grid import IcoGrid
    my_grid = IcoGrid(15)
    Pseudotrajectory("protein0", "NA", my_grid,
                     traj_type="full").generate_pseudotrajectory(initial_distance_nm=1.5,
                                                                 radii=DEFAULT_DISTANCES_PROTEIN)
