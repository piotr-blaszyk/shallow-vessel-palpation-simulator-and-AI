import cv2
import numpy as np



class RigidMpm:
    def __init__(self):
        pass

    @staticmethod
    def colour_mpm_grid_nodes():
        px = 0.105
        py = 0.022
        d = 0.002
        nx0 = px / d
        ny0 = py / d
        nx = int(np.ceil(nx0))+1
        ny = int(np.ceil(ny0))+1
        vx = 0.050
        vy = py
        r0 = 1e-3
        r1 = 1.5*d
        grid_node_positions = np.zeros(shape=(nx, ny, 2), dtype=float)
        for i in range(nx):
            for j in range(ny):
                grid_node_positions[i, j] = np.array([i*d, j*d], dtype=float)
        r = r0 + r1
        r = r0
        distances = np.sqrt(np.sum((grid_node_positions - np.array([vx, vy]))**2, axis=2))
        grid_node_v0_mask = distances <= r
        r_interm_outer = r+1.5*d

        x_min = np.min(grid_node_positions[:, :, 0])
        x_max = np.max(grid_node_positions[:, :, 0])
        y_min = np.min(grid_node_positions[:, :, 1])
        y_max = np.max(grid_node_positions[:, :, 1])
        th = 350
        tw = th * px/py
        scale_factor = th / (y_max - y_min)
        offset = 50
        img_height = int(th + offset*2)
        img_width = int(tw + offset*2)
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        def to_pixel_coords(x, y):
            px = int((x - x_min) * scale_factor + offset)
            py = int((y - y_min) * scale_factor + offset)
            return (px, img_height - py)
        for i in range(grid_node_positions.shape[0]):
            for j in range(grid_node_positions.shape[1]):
                x, y = grid_node_positions[i, j]
                px, py = to_pixel_coords(x, y)
                color = (0, 0, 0) if grid_node_v0_mask[i, j] else (0, 0, 255)
                cv2.circle(img, (px, py), 2, color, -1)
        center_px, center_py = to_pixel_coords(vx, vy)
        cv2.circle(img, (center_px, center_py), 4, (0, 255, 0), -1)
        radius_px_r0 = int(r0 * scale_factor)
        cv2.circle(img, (center_px, center_py), radius_px_r0, (0, 255, 0), 1)
        radius_px_outer = int(r_interm_outer * scale_factor)
        cv2.circle(img, (center_px, center_py), radius_px_outer, (255, 0, 0), 1)
        cv2.imshow('MPM Grid Nodes', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    RigidMpm.colour_mpm_grid_nodes()


if __name__ == '__main__':
    main()
