import logging
import os
from typing import Dict

import numpy as np

from tensorboardX import SummaryWriter

from week_07_tts_am.fastpitch.data import Wav

logger = logging.getLogger(__name__)


class FastpitchLogger:
    def __init__(self, args):
        self.logs_dir = args.logs

        os.makedirs(self.logs_dir, exist_ok=True)

        self.train_logger = SummaryWriter(os.path.join(self.logs_dir, 'train'))
        self.val_logger = SummaryWriter(os.path.join(self.logs_dir, 'val'))
        self.files_logger = SummaryWriter(os.path.join(self.logs_dir, 'files'))

    def log_grads(self, step, model):
        norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        for stat in ('max', 'min', 'mean'):
            self.train_logger.add_scalar(f'grad_{stat}', getattr(np, stat)(norms), step)

    def log(self, logger, step, meta):
        for k, v in meta.items():
            logger.add_scalar(k, v.item() if hasattr(v, 'item') else v, step)

    def log_training(self, step, meta, elapsed_load, elapsed_forward, elapsed_backward):
        elapsed = elapsed_forward + elapsed_backward
        logger.info(f"train : {step}  loss {meta['loss']:.4f}  elapsed {elapsed:.2f}  load {elapsed_load:.2f}")
        self.log(self.train_logger, step, meta)

    def log_validation(self, step, meta):
        logger.info(f"val : {step}  loss {meta['loss']}")
        self.log(self.val_logger, step, meta)

        for w in self.train_logger.all_writers.values():
            w.flush()
        for w in self.val_logger.all_writers.values():
            w.flush()

    def log_images(self, step, images_dict: Dict[str, np.ndarray]):
        logger.info(f"Logging {len(images_dict)} images")
        for image_tag, image in images_dict.items():
            self.files_logger.add_image(
                f'{image_tag}', image, global_step=step,
            )
        self.files_logger.flush()

    def log_audios(self, step, audios_dict: Dict[str, Wav]):
        logger.info(f"Logging {len(audios_dict)} audios")
        for audio_tag, audio in audios_dict.items():
            self.files_logger.add_audio(
                f'{audio_tag}', audio.data, global_step=step, sample_rate=audio.sr
            )
        self.files_logger.flush()

    def log_texts(self, step, texts_dict: Dict[str, str]):
        logger.info(f"Logging {len(texts_dict)} texts")
        for text_tag, text in texts_dict.items():
            self.files_logger.add_text(
                f'{text_tag}', text, global_step=step
            )
        self.files_logger.flush()