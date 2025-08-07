import cadquery as cq
import numpy as np
import math
import pickle
import os
from typing import Dict, Any, Optional
import trimesh


class ViTacTipMeshGenerator:
    """CadQuery-based mesh generator for ViTacTip tactile sensor.
    
    This class replicates the core geometric functionality of the GMSH implementation
    for generating the ViTacTip tactile sensor mesh using CadQuery.
    """
    
    def __init__(self, system_params: Optional[Dict[str, Any]] = None):
        """Initialize the mesh generator.
        
        Args:
            system_params: Dictionary containing system parameters. If None, uses default values.
        """
        self.system_params = system_params or self._get_default_params()
        self.geometry_data = {}
        self.biomimetic_tip_points = None
        self.A_points = None
        self.B_points = None
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default system parameters matching the GMSH implementation."""
        return {
            "vitactip": {
                "number_of_biomimetic_tips": 0,
                "number_of_materials": 2
            },
            "gmsh_mm": {
                "stem_wall_radius_outer": 200.0,  # mm
                "stem_wall_radius_inner": 190.0,  # mm
                "cap_height": 60.0,  # mm
                "stem_height": 300.0,  # mm
                "shell_thickness": 10.0,  # mm
                "dist_eps": 1.0  # mm
            }
        }
    
    def spherical_cap_volume_1(self, radius_of_curvature: float, cap_height: float) -> float:
        """Calculate spherical cap volume using radius of curvature."""
        return 1/3 * math.pi * cap_height**2 * (3 * radius_of_curvature - cap_height)
    
    def spherical_cap_volume_2(self, radius_of_base: float, cap_height: float) -> float:
        """Calculate spherical cap volume using radius of base."""
        return 1/6 * math.pi * cap_height * (3 * radius_of_base**2 + cap_height**2)
    
    def cylinder_volume(self, radius_of_base: float, height: float) -> float:
        """Calculate cylinder volume."""
        return math.pi * radius_of_base**2 * height
    
    def dome_radius_of_curvature(self, radius_of_projection: float, height: float) -> float:
        """Calculate radius of curvature for a dome."""
        r = radius_of_projection
        h = height
        return (r**2 + h**2) / (2 * h)
    
    def calculate_volumes_SI(self) -> Dict[str, float]:
        """Calculate volumes in SI units (m³)."""
        stem_wall_radius_outer = self.geometry_data["stem_wall_radius_outer"]
        stem_wall_radius_inner = self.geometry_data["stem_wall_radius_inner"]
        stem_height = self.geometry_data["stem_height"]
        cap_height = self.geometry_data["cap_height"]
        
        outer_cylinder = self.cylinder_volume(stem_wall_radius_outer, stem_height)
        outer_cap = self.spherical_cap_volume_2(stem_wall_radius_outer, cap_height)
        outer_solid_volume = outer_cylinder + outer_cap
        
        inner_cylinder = self.cylinder_volume(stem_wall_radius_inner, stem_height)
        inner_cap = self.spherical_cap_volume_2(stem_wall_radius_inner, cap_height)
        inner_solid_volume = inner_cylinder + inner_cap
        
        shell_volume = outer_solid_volume - inner_solid_volume
        gel_volume = inner_solid_volume
        
        # Convert from mm³ to m³
        shell_volume /= 1e3**3
        gel_volume /= 1e3**3
        
        volumes = {
            "shell": shell_volume,
            "gel": gel_volume,
            "all": outer_solid_volume / 1e3**3,
        }
        return volumes
    
    def load_parameters(self):
        """Load biomimetic tip parameters if available."""
        # This would load from a pickle file in the original implementation
        # For now, we'll create dummy data or skip if file doesn't exist
        try:
            # Try to load from the expected file path
            biomimetic_tip_file = "difftactile/output/biomimetic-tip-points.pkl"
            if os.path.exists(biomimetic_tip_file):
                with open(biomimetic_tip_file, "rb") as f:
                    self.biomimetic_tip_points = pickle.load(f)
                self.A_points = self.biomimetic_tip_points["A_points"]
                self.B_points = self.biomimetic_tip_points["B_points"]
                
                # Find closest A point to origin
                distances_A = np.sqrt(self.A_points[:, 0]**2 + self.A_points[:, 1]**2)
                closest_A_idx = np.argmin(distances_A)
                self.A_points = self.A_points[closest_A_idx:closest_A_idx + 1]
                self.B_points = self.B_points[closest_A_idx:closest_A_idx + 1]
        except Exception as e:
            print(f"Could not load biomimetic tip parameters: {e}")
            self.biomimetic_tip_points = None
            self.A_points = None
            self.B_points = None
    
    def setup_geometry(self):
        """Setup geometry parameters."""
        stem_wall_radius_outer = self.system_params["gmsh_mm"]["stem_wall_radius_outer"]
        stem_wall_radius_inner = self.system_params["gmsh_mm"]["stem_wall_radius_inner"]
        cap_height = self.system_params["gmsh_mm"]["cap_height"]
        stem_height = self.system_params["gmsh_mm"]["stem_height"]
        
        radius_of_curvature_outer = self.dome_radius_of_curvature(
            stem_wall_radius_outer, cap_height
        )
        radius_of_curvature_inner = radius_of_curvature_outer - self.system_params["gmsh_mm"]["shell_thickness"]
        z_cap_base = radius_of_curvature_outer - cap_height
        z_bottom = z_cap_base - stem_height
        
        self.geometry_data = {
            "stem_wall_radius_outer": stem_wall_radius_outer,
            "stem_wall_radius_inner": stem_wall_radius_inner,
            "cap_height": cap_height,
            "stem_height": stem_height,
            "radius_of_curvature_outer": radius_of_curvature_outer,
            "radius_of_curvature_inner": radius_of_curvature_inner,
            "z_cap_base": z_cap_base,
            "z_bottom": z_bottom,
        }
    
    def create_spherical_cap(self, radius: float, height: float, center: tuple = (0, 0, 0)) -> cq.Workplane:
        """Create a spherical cap using CadQuery."""
        # Create a sphere and cut it with a plane to get the cap
        sphere = cq.Workplane("XY").sphere(radius)
        
        # Cut the sphere with a plane at the cap height
        cut_plane = cq.Workplane("XY").workplane(offset=center[2] + radius - height)
        cap = sphere.split(cut_plane)
        
        # Keep the bottom part (the cap)
        return cap.solids().val()
    
    def create_cylinder(self, radius: float, height: float, center: tuple = (0, 0, 0)) -> cq.Workplane:
        """Create a cylinder using CadQuery."""
        return cq.Workplane("XY").workplane(offset=center[2]).cylinder(height, radius)
    
    def create_biomimetic_tip(self, A_point: np.ndarray, B_point: np.ndarray, radius: float = 0.25) -> cq.Workplane:
        """Create a biomimetic tip cylinder between points A and B."""
        AB = B_point - A_point
        length = np.linalg.norm(AB)
        
        # Create cylinder along the AB direction
        tip = cq.Workplane("XY").workplane(offset=A_point[2]).cylinder(
            length, radius, 
            combine=False, 
            clean=True
        )
        
        # Rotate to align with AB direction
        if length > 0:
            direction = AB / length
            # Calculate rotation angles
            # This is a simplified rotation - in practice you'd need more complex rotation logic
            tip = tip.rotate((0, 0, 0), (0, 0, 1), 0)  # Placeholder rotation
        
        return tip
    
    def generate_vitactip_mesh(self) -> Dict[str, Any]:
        """Generate the ViTacTip mesh using CadQuery."""
        self.load_parameters()
        self.setup_geometry()
        
        number_of_materials = self.system_params["vitactip"]["number_of_materials"]
        number_of_biomimetic_tips = self.system_params["vitactip"]["number_of_biomimetic_tips"]
        
        # Extract geometry parameters
        radius_of_curvature_outer = self.geometry_data["radius_of_curvature_outer"]
        radius_of_curvature_inner = self.geometry_data["radius_of_curvature_inner"]
        z_cap_base = self.geometry_data["z_cap_base"]
        z_bottom = self.geometry_data["z_bottom"]
        stem_wall_radius_outer = self.geometry_data["stem_wall_radius_outer"]
        stem_height = self.geometry_data["stem_height"]
        
        if number_of_materials == 1:
            # Single material case
            # Create outer sphere
            outer_sphere = cq.Workplane("XY").sphere(radius_of_curvature_outer)
            
            # Create cylinder helper to cut the sphere
            cyl_helper = cq.Workplane("XY").workplane(offset=z_bottom).cylinder(
                stem_height * 2, stem_wall_radius_outer
            )
            
            # Intersect sphere with cylinder
            all_volume = outer_sphere.intersect(cyl_helper)
            
            result = {
                "all_volume": all_volume,
                "shell": None,
                "gel": None
            }
            
        elif number_of_materials == 2:
            # Two materials case (shell + gel)
            # Create outer sphere
            outer_sphere = cq.Workplane("XY").sphere(radius_of_curvature_outer)
            
            # Create inner sphere
            inner_sphere = cq.Workplane("XY").sphere(radius_of_curvature_inner)
            
            # Create shell by subtracting inner sphere from outer sphere
            shell = outer_sphere.cut(inner_sphere)
            
            # Create cylinder helper to cut the shell
            cyl_helper = cq.Workplane("XY").workplane(offset=z_bottom).cylinder(
                stem_height * 2, stem_wall_radius_outer
            )
            
            # Intersect shell with cylinder
            shell = shell.intersect(cyl_helper)
            
            # Add biomimetic tips if specified
            if number_of_biomimetic_tips == 1 and self.A_points is not None:
                A = self.A_points[0]
                B = self.B_points[0]
                tip_cylinder = self.create_biomimetic_tip(A, B)
                
                # Fragment the shell with the tip
                # Note: CadQuery doesn't have direct fragment operation like GMSH
                # We'll use a boolean operation instead
                shell = shell.cut(tip_cylinder)
            
            # Create stem wall
            stem_wall_outer = cq.Workplane("XY").workplane(offset=z_bottom).cylinder(
                stem_height, stem_wall_radius_outer
            )
            stem_wall_inner = cq.Workplane("XY").workplane(offset=z_bottom).cylinder(
                stem_height, self.geometry_data["stem_wall_radius_inner"]
            )
            
            # Create stem wall by subtracting inner cylinder from outer cylinder
            stem_wall = stem_wall_outer.cut(stem_wall_inner)
            
            # Combine shell with stem wall
            shell = shell.union(stem_wall)
            
            # Create the complete outer volume
            outer_sphere_complete = cq.Workplane("XY").sphere(radius_of_curvature_outer)
            cyl_helper_complete = cq.Workplane("XY").workplane(offset=z_bottom).cylinder(
                stem_height * 2, stem_wall_radius_outer
            )
            all_volume = outer_sphere_complete.intersect(cyl_helper_complete)
            
            # Create gel by subtracting shell from all volume
            gel = all_volume.cut(shell)
            
            result = {
                "all_volume": all_volume,
                "shell": shell,
                "gel": gel
            }
            
        else:
            raise ValueError("number_of_materials must be 1 or 2")
        
        # Calculate volumes
        volumes = self.calculate_volumes_SI()
        
        # Print volume information
        if number_of_materials == 1:
            print(f"Volume: {volumes['all']:0.3e} m³")
        else:
            print(f"Gel Volume: {volumes['gel']:0.3e} m³")
            print(f"Shell Volume: {volumes['shell']:0.3e} m³")
        
        return {
            "result": result,
            "volumes": volumes,
            "geometry_data": self.geometry_data
        }
    
    def export_mesh(self, result: Dict[str, Any], output_file: str = "vitactip_mesh.stl"):
        """Export the mesh to STL file."""
        if result["result"]["gel"] is not None:
            # Export the main volume
            cq.exporters.export(result["result"]["gel"], output_file)
            print(f"Mesh exported to {output_file}")
        else:
            print("No mesh to export")


def main():
    """Main function to demonstrate the CadQuery implementation."""
    # Create mesh generator
    generator = ViTacTipMeshGenerator()
    
    # Generate mesh
    result = generator.generate_vitactip_mesh()
    
    # Export mesh
    generator.export_mesh(result, "vitactip_cadquery.stl")
    
    print("ViTacTip mesh generation completed using CadQuery!")


if __name__ == "__main__":
    main() 