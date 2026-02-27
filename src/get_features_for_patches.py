import os
import glob
import argparse
import pandas as pd
import numpy as np
from utils import save_features_with_names
from feature_extractor import extractFeatures
from huggingface_hub import login
import time
from PIL import Image

   
FM_NAMES = ['uni', 'gigapath', 'virchow']

def main(input_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/patches',
        output_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/features',
        train=False, batch_size=256):

    print(f"{time.strftime('%H:%M')} starting get_features",flush=True)
    
    #login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    login(token="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", add_to_git_credential=False)

    print(f"{time.strftime('%H:%M')} about to start for loop",flush=True)
    for model in FM_NAMES:

        if train:
            segments=['CD45+','PanCK+']
        else:
            segments = ['']
        

        for segment in segments:
            print(f"{time.strftime('%H:%M')} FM: {model} | segment: {segment}",flush=True)

            patch_paths = glob.glob(os.path.join(input_path,segment,'*.png'))

            #print(patch_paths)
            assert patch_paths, "No patches in the input folder"
            print(f"{time.strftime('%H:%M')} found {len(patch_paths)} patches", flush=True)

            all_features = []
            # process in batches
            for start in range(0, len(patch_paths), batch_size):
                end = start + batch_size
                batch_paths = patch_paths[start:end]

                # load only this batch into memory
                patches = []
                for path in batch_paths:
                    with Image.open(path) as img:
                        # convert/copy so file handle can close
                        patches.append(img.convert("RGB").copy())

                print(f"{time.strftime('%H:%M')} extracting features for patches {start}:{end}", flush=True)
                batch_features = extractFeatures(model, patches)  # expect shape (batch_size, n_features)

                # free image batch ASAP
                del patches

                # move features to CPU / plain numpy if they’re e.g. torch tensors
                #if hasattr(batch_features, "detach"):
                #    batch_features = batch_features.detach().cpu().numpy()
                #elif hasattr(batch_features, "cpu"):
                #    batch_features = batch_features.cpu().numpy()

                all_features.append(batch_features)

            # concatenate all batches
            feature_matrix = np.concatenate(all_features, axis=0)
            del all_features

            print(f"{time.strftime('%H:%M')} saving features", flush=True)
            save_features_with_names(
                feature_matrix,
                patch_paths,
                output_path,
                f"{model}-{segment}",
                format="csv",
            )

            # last bit of housekeeping in case of GPU models
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass



            #patches = [Image.open(path) for path in patch_paths]

            # LOAD IMAGES WITHOUT LEAKING FILE HANDLES
            #patches = []
            #for path in patch_paths:
            #    with Image.open(path) as img:
            #        # copy into memory so the file can be closed
            #        patches.append(img.copy())

            #print(f"getting features for {len(patch_paths)} patchs")
            #feature_matrix = extractFeatures(model, patches)
            #print(f"{time.strftime('%H:%M')} saving features",flush=True)
            #save_features_with_names(feature_matrix, patch_paths, output_path, f"{model}-{segment}", format="csv")



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to h&e patches or path to patches')
    ap.add_argument('-op', '--output_path', required=True, help='path to save output feature file')
    ap.add_argument('-t', '--train', action='store_true')

    args = ap.parse_args()

    #input_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/patches'
    #output_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/features'
    print("train:",args.train)
    main(args.input_path,args.output_path, args.train) 
