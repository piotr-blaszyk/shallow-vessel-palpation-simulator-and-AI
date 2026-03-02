from PIL import Image
import numpy as np


INPUT_IMAGE = "/home/psb120/Documents/diff-tactile-fork/difftactile/output/confusion_overlay_vein_map.png"
OUTPUT_IMAGE = "/home/psb120/Documents/diff-tactile-fork/difftactile/output/confusion_overlay_vein_map_pretty.png"


def main() -> None:
	image = Image.open(INPUT_IMAGE).convert("RGBA")
	pixels = np.array(image)

	white_mask = (
		(pixels[:, :, 0] == 255)
		& (pixels[:, :, 1] == 255)
		& (pixels[:, :, 2] == 255)
	)

	pixels[white_mask, 0] = 0
	pixels[white_mask, 1] = 255
	pixels[white_mask, 2] = 0

	Image.fromarray(pixels, mode="RGBA").save(OUTPUT_IMAGE)
	print(f"Converted {int(white_mask.sum())} white pixels to green.")
	print(f"Saved output to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
	main()
