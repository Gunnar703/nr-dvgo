python run.py --config configs/custom/beam.py
python run.py --config configs/custom/beam.py --export_bbox_and_cams_only beam_cams.npz
python run.py --config configs/custom/beam.py --export_coarse_only beam_geom.npz
python run.py --config configs/custom/beam.py --render_only --render_test
python tools/vis_volume.py beam_geom.npz 0.001 --cam beam_cams.npz
