import dataclasses
import glob
import logging
import os

import torch

logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def get_last_checkpoint_path(self):
        checkpoint_list = glob.glob(os.path.join(self.checkpoint_dir, 'fp_step_*'))
        if not checkpoint_list:
            return ''
        return sorted(checkpoint_list)[-1]

    def save_checkpoint(self, model, optimizer, step, epoch):
        fp_ckpt = {
            'step': step,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'hparams': dataclasses.asdict(model.hparams),
            'optimizer_state_dict':  optimizer.state_dict()
        }
        prev_checkpoint = self.get_last_checkpoint_path()

        ckpt_name = f"fp_step_{step:08d}.pt"
        fpath = os.path.join(self.checkpoint_dir, ckpt_name)
        logger.info(f"Saving model state at step {step} to {fpath}")
        torch.save(fp_ckpt, fpath)

        if os.path.exists(prev_checkpoint):
            os.remove(prev_checkpoint)

    def load_last_checkpoint(self):
        last_ckpt_path = self.get_last_checkpoint_path()
        if not last_ckpt_path:
            return {'step': 0, 'epoch': 0}

        last_ckpt = torch.load(last_ckpt_path, map_location='cpu')
        logger.info(f"Loaded {last_ckpt_path}")
        return last_ckpt
