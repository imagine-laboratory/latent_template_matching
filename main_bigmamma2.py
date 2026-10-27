# main_bigmamma2.py
"""
Entry point for BigMamma2 experiments.
Run either:
  python main_bigmamma2.py extract <args for latent_template_extract>
  python main_bigmamma2.py count   <args for count_from_latent>
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="BigMamma2: Latent template experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # We don’t define all args here; we just select which tool to run.
    subparsers.add_parser("extract", help="Extract latent template with RoIAlign")
    subparsers.add_parser("count", help="Count pineapples using latent template")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    cmd = sys.argv[1]
    forward_args = sys.argv[2:]  # everything after extract/count

    if cmd == "extract":
        from tools import latent_template_extract
        sys.argv = ["latent_template_extract.py"] + forward_args
        latent_template_extract.main()

    elif cmd == "count":
        from experiments import count_from_latent
        sys.argv = ["count_from_latent.py"] + forward_args
        count_from_latent.main()

if __name__ == "__main__":
    main()



#python main_bigmamma2.py extract --images_dir /data/ffallas/datasets/vae/FULL_UNIFIED --csv /data/ffallas/datasets/vae/FULL_UNIFIED_labels.csv --checkpoint /data/ffallas/generative/VAE/output/checkpoints/betaKL@0.001/weights_ck_397.pt --output_dir /data/ffallas/generative/VAE/template_bank --max_crops 1 --roi_size 9 --resize_img 256 --feature_source feats
    
#python main_bigmamma2.py count --images_dir /data/ffallas/datasets/vae/FULL_UNIFIED --prototype_pt /data/ffallas/generative/VAE/template_bank/latent_prototype.pt --checkpoint /data/ffallas/generative/VAE/output/checkpoints/betaKL@0.001/weights_ck_397.pt --output_dir /data/ffallas/generative/VAE/count_outputs_single --resize_img 256 --thresh 0.6 --pool_ks 3 --use_prototype_image --csv /data/ffallas/datasets/vae/FULL_UNIFIED_labels.csv --feature_source auto
