import os, sys
from PIL import Image, ImageDraw
import argparse
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--indir",required=True)
	parser.add_argument("--outdir",required=True)
	args=parser.parse_args()

	paths = os.listdir(args.indir)

	i = 1
	for path in paths:
		print(path)
		if "pdf" in path:

			newdir = f"bal{i}"
			os.mkdir(f"{args.outdir}/{newdir}")
			images = convert_from_path(f"{args.indir}/{path}")
			j=0
			for img in images:
				img.save(f"{args.outdir}/{newdir}/balimg{i}_{j}.jpg")
				j+=1

			i+=1
	print("done.")