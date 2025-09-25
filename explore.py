import argparse
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt

def read_image_rgb(image_path: Path) -> Tuple[cv2.Mat, str]:
	"""Read an image with OpenCV and convert BGR to RGB if needed.

	Returns the numpy array and a short description for the title.
	"""
	image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
	if image is None:
		raise FileNotFoundError(f"Failed to read image: {image_path}")
	desc = ""
	if len(image.shape) == 2:
		desc = "grayscale"
	elif len(image.shape) == 3 and image.shape[2] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		desc = "RGB"
	elif len(image.shape) == 3 and image.shape[2] == 4:
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
		desc = "RGBA"
	else:
		desc = f"shape {image.shape}"

	return image, desc


def read_mask_grayscale(mask_path: Path) -> cv2.Mat:
	"""Read mask as a single-channel (grayscale) image."""
	mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
	if mask is None:
		raise FileNotFoundError(f"Failed to read mask: {mask_path}")
	return mask


def show_side_by_side(image, mask, image_title: str, mask_title: str, mask_cmap: str) -> None:
	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	ax_img, ax_mask = axes

	ax_img.imshow(image, cmap="gray" if image.ndim == 2 else None)
	ax_img.set_title(image_title)
	ax_img.axis("off")

	ax_mask.imshow(mask, cmap=mask_cmap, interpolation="nearest")
	ax_mask.set_title(mask_title)
	ax_mask.axis("off")

	plt.tight_layout()
	plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Display an image and a mask side-by-side using OpenCV + Matplotlib",
	)
	parser.add_argument(
		"--image_path",
		required=True,
		type=Path,
		help="Path to the image file (e.g. dataset/images/img.png)",
	)
	parser.add_argument(
		"--mask_path",
		required=True,
		type=Path,
		help="Path to the mask file (e.g. dataset/masks/img_mask.png)",
	)
	parser.add_argument(
		"--mask_cmap",
		default="gray",
		choices=[
			"gray",
			"viridis",
			"magma",
			"plasma",
			"inferno",
			"turbo",
		],
		help="Matplotlib colormap for the mask display",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	image, img_desc = read_image_rgb(args.image_path)
	mask = read_mask_grayscale(args.mask_path)

	image_title = f"Image: {args.image_path.name} ({img_desc})"
	mask_title = f"Mask: {args.mask_path.name} (grayscale)"
	show_side_by_side(image, mask, image_title, mask_title, args.mask_cmap)


if __name__ == "__main__":
	main()