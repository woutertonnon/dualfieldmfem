import os
import shutil
import gmsh
import glob

output_dir = "../3d"

# Delete the directory and its contents if it already exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a fresh, empty directory
os.makedirs(output_dir, exist_ok=True)

for geo_file in glob.glob("*.geo"):
    print(f"Meshing {geo_file}...")
    
    gmsh.initialize()
    gmsh.open(geo_file)
    
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Algorithm", 1)   # MeshAdapt for 2D
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    
    gmsh.model.mesh.generate(3)
    
    base_name = os.path.basename(geo_file)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, name_without_ext + ".msh")
    
    gmsh.write(output_path)
    print(f"Saved to: {output_path}")
    
    gmsh.finalize()