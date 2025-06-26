import open3d as o3d
import numpy as np

def create_3d_surface_mesh(grid_size=10, scale_z=0.5):
    """
    Creates a 3D surface mesh (e.g., a "hill" or "bowl" shape) by
    generating a grid of points and adjusting their Z-coordinates.
    """
    vertices = []
    triangles = []
    
    # 1. Generate vertices in a grid
    # We'll create a grid_size x grid_size grid of points
    for i in range(grid_size):
        for j in range(grid_size):
            x = i / (grid_size - 1) * 2.0 - 1.0  # Normalize x to [-1, 1]
            y = j / (grid_size - 1) * 2.0 - 1.0  # Normalize y to [-1, 1]
            
            # Create a simple "hill" or "bowl" shape using a sine wave or parabolic function
            # For a hill:
            z = np.exp(-(x**2 + y**2) * 3) * scale_z # Gaussian-like hill
            # For a more complex surface, you could use:
            # z = np.sin(x * np.pi * 3) * np.cos(y * np.pi * 2) * scale_z
            
            vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float64)

    # 2. Generate triangles (quads to two triangles)
    # This part connects the grid points to form triangles.
    # Each quad (4 points) in the grid will be split into two triangles.
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Get the indices of the 4 corner points of the current quad
            idx00 = i * grid_size + j
            idx10 = (i + 1) * grid_size + j
            idx01 = i * grid_size + (j + 1)
            idx11 = (i + 1) * grid_size + (j + 1)

            # Create two triangles for the quad
            # Triangle 1: (0,0) -> (1,0) -> (0,1)
            triangles.append([idx00, idx10, idx01])
            # Triangle 2: (1,0) -> (1,1) -> (0,1)
            triangles.append([idx10, idx11, idx01])

    triangles = np.array(triangles, dtype=np.int32)

    # Create an Open3D TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Compute vertex normals for proper lighting/shading
    mesh.compute_vertex_normals()

    # Optional: Assign a color based on Z-value for better visualization
    # This makes higher points one color and lower points another
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    
    colors = np.zeros_like(vertices)
    for k in range(len(vertices)):
        normalized_z = (vertices[k, 2] - min_z) / (max_z - min_z)
        # Simple color gradient from blue (low) to red (high)
        colors[k, 0] = normalized_z  # Red channel (higher Z -> more red)
        colors[k, 2] = 1 - normalized_z # Blue channel (lower Z -> more blue)
        # Green channel can be constant or also vary
        colors[k, 1] = 0.2 # A little green for more vibrant colors

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh

def visualize_mesh(mesh):
    """
    Visualizes an Open3D mesh.
    """
    print("Displaying 3D surface mesh. Press 'H' for help with controls.")
    o3d.visualization.draw_geometries([mesh],
                                      mesh_show_wireframe=False, # Often cleaner without wireframe for smooth surfaces
                                      mesh_show_back_face=True)

if __name__ == "__main__":
    # Create the 3D surface mesh (e.g., a "hill")
    hill_mesh = create_3d_surface_mesh(grid_size=50, scale_z=0.3)

    # Visualize the mesh
    visualize_mesh(hill_mesh)

    # Example of loading a standard 3D mesh (e.g., dragon)
    # Ensure you have the Open3D examples data downloaded or replace with your own path
    # try:
    #     dragon_mesh = o3d.data.Dragon().path
    #     mesh_load = o3d.io.read_triangle_mesh(dragon_mesh)
    #     mesh_load.compute_vertex_normals()
    #     print("Displaying loaded dragon mesh.")
    #     visualize_mesh(mesh_load)
    # except Exception as e:
    #     print(f"Could not load example dragon mesh: {e}. Make sure Open3D example data is available.")