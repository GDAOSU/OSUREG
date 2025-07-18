import os
import numpy as np
import subprocess
import sys


DSM_ICP_EXE = './exe/reg.exe'
DSM_TRANSFORM_EXE = './exe/transform.exe'
# DSM_EVAL_EXE = './exe/dsm_eval.exe'

def dsm_reg_exe(src_dsm_path, ref_dsm_path, out_dir, progress=None, reg=None):
    import shutil
    ICP_CORR_THRESH = 10
    ICP_NUM_ITERS = 100
    result_path = os.path.join(out_dir, os.path.basename(src_dsm_path)[:-4] + "_" + os.path.basename(ref_dsm_path)[
                                                                                    :-4] + "_reg.txt")

    cmd = [DSM_ICP_EXE, '-src', src_dsm_path, '-dst', ref_dsm_path, '-outdir', out_dir, '-outlier_thresh',
           str(ICP_CORR_THRESH), '-icp_max_iter', str(ICP_NUM_ITERS), '-gen_data', '1']
    subprocess.run(cmd)
    if progress:
        progress.advance(reg, 90)

    src_par_dir = os.path.dirname(src_dsm_path)
    src_reg_tif = src_dsm_path[:-4] + "_reg.tif"
    src_reg_tfw = src_dsm_path[:-4] + "_reg.tfw"
    out_tif = src_reg_tif.replace(src_par_dir, out_dir)
    out_tfw = src_reg_tfw.replace(src_par_dir, out_dir)
    shutil.copyfile(src_reg_tif, out_tif)
    shutil.copyfile(src_reg_tfw, out_tfw)
    os.remove(src_reg_tfw)
    os.remove(src_reg_tif)
    print("Registration finished! The result is written to {}".format(out_tif))


if __name__ == '__main__':
    dsm_reg_exe(r'E:\data\tmp\ma\sat-lidar\sat.tif', r'E:\data\tmp\ma\sat-lidar\lidar.tif', r'E:\data\tmp\ma\sat-lidar')
