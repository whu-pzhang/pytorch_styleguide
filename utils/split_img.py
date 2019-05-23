import numpy as np
import cv2
from pathlib import Path
import rasterio

__all__ = ["split_img"]


def split_img(raster_file, outpath=None, patch_size=1000, overlap_ratio=0.1, start_index=1, verbose=True):
    splitbase = SplitBase(raster_file, outpath, patch_size, overlap_ratio, start_index, verbose)
    splitbase.split_img(shuffix="jpg", keep_thres=0.3)


class SplitBase():
    def __init__(self,
                 raster_file,
                 outpath=None,
                 patch_size=1000,
                 overlap_ratio=0.1,
                 start_index=1,
                 verbose=True
                 ):
        self.raster_file = Path(raster_file)
        if outpath is None:
            self.outpath = Path(raster_file).parent / "cache"
            self.outpath.mkdir(parents=True, exist_ok=True)
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.start_index = start_index
        self.verbose = verbose

        self.slide_step = int(patch_size * (1 - overlap_ratio))
        self.dataset = rasterio.open(raster_file)
        self.raster_width = self.dataset.width
        self.raster_height = self.dataset.height
        # self.num_bands = self.dataset.count
        # self.geotransform = self.dataset.transform

        if self.verbose:
            print("="*80)
            print("[INFO]")
            print(f"Rasterfile:  {self.raster_file}")
            print(f"Output path: {self.outpath}")
            print(f"Patch size:  {self.patch_size}")
            print(f"Start index: {self.start_index}")
            print("="*80)

    def split_img(self, shuffix="jpg", keep_thres=0.3):
        """[summary]
        
        Args:
            shuffix (str, optional): the image format. Defaults to "jpg".
            keep_thres (float, optional): threshold to determite weather to keep the patch image. Defaults to 0.3.
        """
        N = self.patch_size
        idx = 0
        for h in range(0, self.raster_height, self.slide_step):
            for w in range(0, self.raster_width, self.slide_step):
                # image patch window
                window = rasterio.windows.Window(w, h, N, N)
                patch = self.dataset.read(window=window)
                if min(patch.shape[1:]) < keep_thres * N:
                    continue
                # depth = patch.shape[0]
                patch_name = f"P{self.start_index + idx:06d}_{w}_{h}"
                self.save_image_patch(patch, patch_name, shuffix)
                idx += 1

    
    def save_image_patch(self, img, patch_name, shuffix):
        img = img.transpose(1, 2, 0).copy()
        patch = img[:, :, ::-1]  # RGB -> BGR
        out_file = self.outpath / f"{patch_name}.{shuffix}"
        cv2.imwrite(str(out_file), patch)
        if self.verbose:
            print(out_file)
