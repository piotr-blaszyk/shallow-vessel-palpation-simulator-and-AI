"""
Script to run fisheye model operations
"""

from difftactile.sensor_model.fisheye_model import FisheyeModel

def main():
    fisheye_model = FisheyeModel()
    # fisheye_model.interactive_exploration()
    # fisheye_model.save_init_marker_positions()
    # fisheye_model.extract_experimental_markers_and_save_to_file()
    fisheye_model.generate_marker_3d_projection()

if __name__ == '__main__':
    main() 