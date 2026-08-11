from PIL import Image
import numpy as np

from difftactile.main.paths import repo_path


INPUT_IMAGE = repo_path("difftactile/output/confusion_overlay_vein_map.png")
OUTPUT_IMAGE = repo_path("difftactile/output/confusion_overlay_vein_map_pretty.png")


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
