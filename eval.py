"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-r', '--replay', default=False)
@click.option('-l', '--close_loop',default=False)
def main(checkpoint, output_dir, device,replay,close_loop):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    # import pdb; pdb.set_trace()
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    # print("cls",cls)

    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace # 指定类的继承
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    # print(policy)
    if cfg.training.use_ema:
        # print("true")
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval() # 推理
    
    
    # run eval
    if replay == True:
        cfg.task.env_runner['_target_'] = 'replay_pusht_video.PushTKeypointsRunner_Replay'
    if close_loop == True:
        cfg.task.env_runner['_target_'] = 'closeloop_pusht_speedup.PushTKeypointsRunner_closeloop_speed'
    # print("cfg.task.env_runner",cfg.task.env_runner)
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    # print("env_runner",env_runner)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
