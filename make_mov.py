from bsl import viz
import sys
from pathlib import Path
case_name=sys.argv[1]
imgs = sorted(Path('.').glob('*.png'))
viz.images_to_movie(imgs, 'fixed_points.mp4'.format(case_name), fps=30)