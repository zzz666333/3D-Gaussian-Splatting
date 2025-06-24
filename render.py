#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    print(f"即将渲染 {name}，视角数：{len(views)}")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        try:
            print(f"渲染第 {idx} 个视角")
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
            gt = view.original_image[0:3, :, :]

            # if args.train_test_exp:
            #     rendering = rendering[..., rendering.shape[-1] // 2:]
            #     gt = gt[..., gt.shape[-1] // 2:]

            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            if hasattr(view, 'original_image') and view.original_image is not None:
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        except Exception as e:
            print(f"渲染第 {idx} 个视角时出错: {e}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, novel_json=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, novel_json=novel_json)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--novel_json", type=str, default=None, help="指定 novel 轨迹的 transforms_xxx.json 路径")
    args = get_combined_args(parser)
    #model_path = model.extract(args).model_path
    #print("Rendering", model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 正确提取参数对象，只 extract 一次
    model_params = model.extract(args)
    pipeline_params = pipeline.extract(args)

    # 修复属性名，确保和 Scene 里用到的完全一致
    if not hasattr(model_params, "model_path") and hasattr(model_params, "_model_path"):
        model_params.model_path = model_params._model_path
    if not hasattr(model_params, "images") and hasattr(model_params, "_images"):
        model_params.images = model_params._images
    if not hasattr(model_params, "depths") and hasattr(model_params, "_depths"):
        model_params.depths = model_params._depths
    if not hasattr(model_params, "white_background") and hasattr(model_params, "_white_background"):
        model_params.white_background = model_params._white_background

    print("model_params:", vars(model_params))

    novel_json = getattr(args, "novel_json", None)
    render_sets(model_params, args.iteration, pipeline_params, args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, novel_json=novel_json)