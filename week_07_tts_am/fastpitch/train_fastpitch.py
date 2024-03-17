# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import json
import logging
import sys
import time

import numpy as np
import soundfile as sf
import torch

from week_07_tts_am.fastpitch.common.checkpointer import Checkpointer
from week_07_tts_am.fastpitch.common.plotting_utils import plot_spectrogram_to_numpy
from week_07_tts_am.fastpitch.common.whisper import calculate_wer
from week_07_tts_am.fastpitch.data import MelSpectrogram, Wav, prepare_loaders
from week_07_tts_am.fastpitch.hparams import HParamsFastpitch
from week_07_tts_am.fastpitch.logger import FastpitchLogger
from week_07_tts_am.fastpitch.loss_function import FastPitchLoss
from week_07_tts_am.fastpitch.model import FastPitch
from week_07_tts_am.fastpitch.validation import generate_predictions, validate
from week_07_tts_am.hifigan.model import load_model as load_hfg_model

logger = logging.getLogger(__name__)

sample_rate = MelSpectrogram.sample_rate

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument(
        "--hparams", type=str, default="",
        help="Path to json with parameter overrides"
    )
    parser.add_argument(
        '--logs', type=str, default='logs',
        help='local directory to save tensorboard logs'
    )
    parser.add_argument(
        '--ckptdir', type=str, required=True,
        help='yt directory to save model checkpoints'
    )
    parser.add_argument(
        "--dataset", type=str, default="./",
        help="Path to dataset"
    )
    parser.add_argument(
        "--hfg", type=str, required=False, default="",
        help="HFG checkpoint for evaluation, either local or on yt (full path without /g)"
    )
    return parser


def get_hparams(hparams_path: str) -> HParamsFastpitch:
    if hparams_path != "":
        with open(hparams_path) as f:
            hparams_overrides = json.load(f)
    else:
        hparams_overrides = {}

    hparams = HParamsFastpitch.create(hparams_overrides)
    return hparams


def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1.0 / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group["lr"] = learning_rate * scale


def train(args):
    hparams = get_hparams(args.hparams)

    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    train_loader, valid_loader = prepare_loaders(args.dataset, hparams)

    model = FastPitch(hparams).to(device)

    if hparams.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
    else:     # The original implementation uses LAMB with betas=(0.9, 0.98), eps=1e-9
        raise ValueError(f'unknown optimizer - {hparams.optimizer}')

    model_logger = FastpitchLogger(args)

    checkpointer = Checkpointer(args.ckptdir)
    fp_ckpt = checkpointer.load_last_checkpoint()

    step = fp_ckpt['step']
    epoch = fp_ckpt['epoch']

    if 'state_dict' in fp_ckpt:
        model.load_state_dict(fp_ckpt['state_dict'])
    if 'optimizer_state_dict' in fp_ckpt:
        optimizer.load_state_dict(fp_ckpt['optimizer_state_dict'])

    hfg = load_hfg_model(args.hfg)
    hfg = hfg.to(device).eval()

    criterion = FastPitchLoss(hparams)

    model.train()

    while True:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        logger.info(f"Epoch num: {epoch}")

        timestamp = time.perf_counter()
        for batch in train_loader:
            elapsed_load = time.perf_counter() - timestamp
            timestamp = time.perf_counter()

            batch = batch.to(device)

            adjust_learning_rate(step, optimizer, hparams.learning_rate, hparams.warmup_steps)
            y_pred = model.forward(batch)

            elapsed_forward = time.perf_counter() - timestamp
            timestamp = time.perf_counter()

            model.zero_grad()
            loss, meta = criterion(y_pred, (batch.mels, batch.durations, batch.text_lengths, batch.pitches))

            loss.backward()

            model_logger.log_grads(step, model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            elapsed_backward = time.perf_counter() - timestamp

            step += 1

            if step == 1:
                # Log GT data
                logger.info('Logging GT data')

                valset = valid_loader.dataset
                examples = [row for i, row in enumerate(valset) if i < hparams.max_val_audios]
                gt_texts_dict = {
                    f'gt_text/{i}': row['meta']['normalized_text'] for i, row in enumerate(examples)
                }
                gt_mels_dict = {
                    f'gt_mel/{i}': plot_spectrogram_to_numpy(row['mel']) for i, row in enumerate(examples)
                }
                gt_audios_dict = {
                    f'gt_audio/{i}': Wav(*sf.read(row['meta']['audio_path']))
                    for i, row in enumerate(examples)
                }

                model_logger.log_images(step, gt_mels_dict)
                model_logger.log_audios(step, gt_audios_dict)
                model_logger.log_texts(step, gt_texts_dict)

            meta['lr'] = optimizer.param_groups[0]["lr"]
            model_logger.log_training(step, meta, elapsed_load, elapsed_forward, elapsed_backward)

            if step % hparams.eval_interval == 0 or step == 1:
                logger.info('Running validation')
                loss, meta = validate(
                    model, criterion, valid_loader, device, use_gt_durations=True, use_gt_pitch=True
                )

                asr_texts_dict, pred_mel_dict, pred_audios_dict = generate_predictions(
                    valset=valid_loader.dataset,
                    fp=model,
                    hfg=hfg,
                    max_predictions=hparams.max_val_audios,
                    device=device
                )
                model_logger.log_images(step, pred_mel_dict)
                model_logger.log_audios(step, pred_audios_dict)
                model_logger.log_texts(step, asr_texts_dict)

                meta['wer'] = calculate_wer(
                    [asr_texts_dict[f'asr_output/{i}'] for i in range(len(asr_texts_dict))],
                    [gt_texts_dict[f'gt_text/{i}'] for i in range(len(gt_texts_dict))]
                )
                model_logger.log_validation(step, meta)
                checkpointer.save_checkpoint(model, optimizer, step, epoch)

            if step >= hparams.num_steps:
                exit()

            timestamp = time.perf_counter()

        epoch += 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format='%(process)d %(asctime)s %(levelname)s %(message)s', datefmt='%d %I:%M:%S',
    )
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)

    train(parser.parse_args())