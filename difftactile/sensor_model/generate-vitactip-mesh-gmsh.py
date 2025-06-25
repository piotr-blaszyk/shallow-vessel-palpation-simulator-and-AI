import gmsh
import numpy as np
import math
import sys
import pickle

def dome_radius_of_curvature(radius_of_projection, height):
    r = radius_of_projection
    h = height
    return (r ** 2 + h ** 2) / (2 * h)

def generate_vitactip_mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ViTacTip")

    stem_wall_radius_outer = 20
    stem_wall_radius_inner = 19
    cap_height = 6.0
    stem_height = 11.0
    R_outer = dome_radius_of_curvature(stem_wall_radius_outer, cap_height)
    R_inner = R_outer - 1
    y_cap_base = R_outer - cap_height
    y_bottom = y_cap_base - stem_height

    outer_dome = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
    inner_dome = gmsh.model.occ.addSphere(0, 0, 0, R_inner)
    cap = gmsh.model.occ.cut([(3, outer_dome)], [(3, inner_dome)])[0]
    cyl_helper = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, stem_wall_radius_outer)
    cap = gmsh.model.occ.intersect(cap, [(3, cyl_helper)])[0]
    stem_wall_outer = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, stem_wall_radius_outer)
    stem_wall_inner = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, stem_wall_radius_inner)
    stem_wall = gmsh.model.occ.cut([(3, stem_wall_outer)], [(3, stem_wall_inner)])[0]
    shell = gmsh.model.occ.fuse(cap, stem_wall)[0]

    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(3)
    gmsh.fltk.run()
    gmsh.finalize()

def box_point():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("sandbox")
    gmsh.model.occ.addCone(x=0, y=0, z=0, dx=0, dy=1, dz=0, r1=0, r2=1/3)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(3)
    gmsh.fltk.run()
    gmsh.finalize()

if __name__ == "__main__":
    generate_vitactip_mesh()
