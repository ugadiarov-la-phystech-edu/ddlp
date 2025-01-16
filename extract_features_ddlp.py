import argparse
import collections
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from torchvision import transforms
from tqdm import tqdm

from models import ObjectDynamicsDLP


def create_ddlp(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    action_dim = config['action_dim']
    # data and general
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    animation_horizon = config['animation_horizon']
    batch_size = config['batch_size']
    max_norm = config.get('max_norm', 0.5)
    lr = config['lr']
    num_epochs = config['num_epochs']
    topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot
    eval_epoch_freq = config['eval_epoch_freq']
    weight_decay = config['weight_decay']
    iou_thresh = config['iou_thresh']  # threshold for NMS for plotting bounding boxes
    run_prefix = config['run_prefix']
    if run_prefix == '':
        run_prefix = os.path.splitext(os.path.basename(config_path))[0]

    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    scheduler_gamma = config['scheduler_gamma']
    eval_im_metrics = config['eval_im_metrics']
    cond_steps = config['cond_steps']  # conditional frames for the dynamics module during inference

    # model
    timestep_horizon = config['timestep_horizon']
    kp_range = config['kp_range']
    kp_activation = config['kp_activation']
    enc_channels = config['enc_channels']
    prior_channels = config['prior_channels']
    pad_mode = config['pad_mode']
    n_kp = config['n_kp']  # kp per patch in prior, best to leave at 1
    n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
    n_kp_enc = config['n_kp_enc']  # total posterior kp
    patch_size = config['patch_size']  # prior patch size
    anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size
    mu_scale_prior = config.get('mu_scale_prior', None)
    learned_feature_dim = config['learned_feature_dim']
    bg_learned_feature_dim = config.get('bg_learned_feature_dim', None)
    dropout = config['dropout']
    use_resblock = config['use_resblock']
    use_correlation_heatmaps = config['use_correlation_heatmaps']  # use heatmaps for tracking
    enable_enc_attn = config['enable_enc_attn']  # enable attention between patches in the particle encoder
    filtering_heuristic = config["filtering_heuristic"]  # filtering heuristic to filter prior keypoints
    use_actions = config.get("use_actions", False)  # use action-conditioned dynamics model
    max_beta_coef = config.get("max_beta_coef", 100)

    # optimization
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_dyn = config['beta_dyn']
    beta_rec = config['beta_rec']
    beta_dyn_rec = config['beta_dyn_rec']
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes
    num_static_frames = config['num_static_frames']  # frames for which kl is calculated w.r.t constant prior params
    train_enc_prior = config['train_enc_prior']

    # priors
    sigma = config['sigma']  # std for constant kp prior, leave at 1 for deterministic chamfer-kl
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

    # transformer - PINT
    pint_layers = config['pint_layers']
    pint_heads = config['pint_heads']
    pint_dim = config['pint_dim']
    predict_delta = config['predict_delta']  # dynamics module predicts the delta from previous step
    start_epoch = config['start_dyn_epoch']

    model = ObjectDynamicsDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                              image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                              pad_mode=pad_mode, sigma=sigma, bg_learned_feature_dim=bg_learned_feature_dim,
                              dropout=dropout, patch_size=patch_size, n_kp_enc=n_kp_enc,
                              n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                              anchor_s=anchor_s, use_resblock=use_resblock,
                              timestep_horizon=timestep_horizon, predict_delta=predict_delta,
                              scale_std=scale_std, offset_std=offset_std, obj_on_alpha=obj_on_alpha,
                              obj_on_beta=obj_on_beta, pint_layers=pint_layers, pint_heads=pint_heads,
                              pint_dim=pint_dim, use_correlation_heatmaps=use_correlation_heatmaps,
                              enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic,
                              max_beta_coef=max_beta_coef, action_dim=action_dim, mu_scale_prior=mu_scale_prior)

    return model, config


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    pretrained_epoch = checkpoint['epoch'] + 1
    valid_loss = best_valid_loss = checkpoint['best_valid_loss']
    best_valid_epoch = checkpoint['best_valid_epoch']
    val_lpips = best_val_lpips = checkpoint['best_val_lpips']
    best_val_lpips_epoch = checkpoint['best_val_lpips_epoch']
    print(f"loaded model from checkpoint: {checkpoint_path}")

    return model


def get_fg_representation(dlp_output, timestep_horizon):
    rep = torch.cat((dlp_output['z'], dlp_output['mu_scale'], dlp_output['mu_depth'], dlp_output['mu_features'],
                     dlp_output['obj_on'].unsqueeze(dim=-1)), dim=-1).detach().cpu().numpy()
    return rep.reshape((-1, timestep_horizon + 1, *rep.shape[1:]))[:, :timestep_horizon]


def get_bg_representation(dlp_output, timestep_horizon):
    rep = dlp_output['z_bg'].detach().cpu().numpy()
    return rep.reshape((-1, timestep_horizon + 1, *rep.shape[1:]))[:, :timestep_horizon]


def get_episode_ids(dataset_path, split):
    split_path = os.path.join(dataset_path, split)
    episode_ids = []
    for episode_id in sorted(os.listdir(split_path), key=lambda x: int(x)):
        path = os.path.join(split_path, episode_id)
        if os.path.isdir(path):
            episode_ids.append(episode_id)
        else:
            print(f'{path} is not a directory!')

    return episode_ids


def read_episode_data(dataset_path, split, episode_id, resolution):
    episode_path = os.path.join(dataset_path, split, str(episode_id))
    images = []
    for image_path in sorted(Path(episode_path).glob('*png'), key=lambda x: int(x.stem)):
        img = Image.open(image_path)
        img = img.resize((resolution, resolution))
        img = transforms.ToTensor()(img)[:3]
        images.append(img)

    images = torch.stack(images, dim=0)
    actions = np.load(os.path.join(episode_path, 'actions.npy'))
    rewards = np.load(os.path.join(episode_path, 'rewards.npy'))

    return {'episode_id': episode_id, 'images': images, 'actions': actions, 'rewards': rewards, 'split': split}


def write_episode_data(dataset_path, episode_id, fg_representations, bg_representations, actions, rewards):
    # fg_representations -> episode_length, horizon, n_particles, particle_dim
    # bg_representations -> episode_length, particle_dim
    # actions -> episode_length, horizon, action_dim
    # rewards -> episode_length, 1
    with h5py.File(dataset_path, 'a') as hf:
        grp = hf.create_group(str(episode_id))
        grp.create_dataset('fg_representation', data=fg_representations)
        grp.create_dataset('bg_representation', data=bg_representations)
        grp.create_dataset('actions', data=actions)
        grp.create_dataset('rewards', data=rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset_path', type=str, required=True)
    parser.add_argument('--target_dataset_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--resize_to', type=int, default=128)
    parser.add_argument('--max_read_workers', type=int, default=2)
    parser.add_argument('--max_write_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.target_dataset_path, exist_ok=False)
    episode_ids = collections.deque([(split, episode_id) for split in ('train', 'val') for episode_id in
                                     get_episode_ids(args.source_dataset_path, split)])
    N = len(episode_ids)

    model, config = create_ddlp(args.config_path)
    model = load_checkpoint(model, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()
    model.requires_grad_(False)

    forkserver = multiprocessing.get_context('forkserver')
    read_executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.max_read_workers, mp_context=forkserver)
    write_executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.max_write_workers, mp_context=forkserver)
    read_futures = collections.deque()
    write_futures = collections.deque()

    processed_pbar = tqdm(total=N, position=0, leave=True, desc='Processed episodes')
    written_pbar = tqdm(total=N, position=1, leave=True, desc='Saved episodes')
    while True:
        if len(episode_ids) > 0 and len(read_futures) < 2 * args.max_read_workers:
            for _ in range(min(2, len(episode_ids))):
                split, episode_id = episode_ids.popleft()
                read_futures.append(read_executor.submit(
                    read_episode_data, args.source_dataset_path, split, episode_id, args.resize_to
                ))

        if len(read_futures) == 0:
            break

        # extract data from episode
        data = read_futures.popleft().result()
        x = data['images'].to(args.device)
        x = torch.cat([x[:1].expand((model.timestep_horizon - 1, -1, -1, -1)), x], dim=0)
        x = x.unfold(dimension=0, size=model.timestep_horizon + 1, step=1).permute((0, 4, 1, 2, 3)).contiguous()
        foreground_representations = []
        background_representations = []
        for batch in torch.split(x, args.batch_size):
            dlp_output = model(batch, deterministic=True, x_prior=batch, warmup=False, noisy=False, forward_dyn=True,
                               train_enc_prior=config['train_enc_prior'], num_static_frames=config['num_static_frames'],
                               predict_next=False)
            foreground_representations.append(get_fg_representation(dlp_output, model.timestep_horizon))
            background_representations.append(get_bg_representation(dlp_output, model.timestep_horizon))

        foreground_representations = np.concatenate(foreground_representations)
        background_representations = np.concatenate(background_representations)
        actions = data['actions']
        actions = np.concatenate(
            [np.zeros((model.timestep_horizon - 1, *actions.shape[1:]), dtype=np.float32,), actions],
            axis=0
        )
        actions = sliding_window_view(actions, window_shape=model.timestep_horizon, axis=0).transpose((0, 2, 1))
        write_futures.append(write_executor.submit(
            write_episode_data,
            os.path.join(args.target_dataset_path, f'{data["split"]}.hdf5'),
            data['episode_id'],
            foreground_representations,
            background_representations,
            actions,
            data['rewards'],
        ))

        processed_pbar.update(1)

        while len(write_futures) > 0 and write_futures[0].done():
            write_futures.popleft().result()
            written_pbar.update(1)

    for write_futures in write_futures:
        write_futures.result()
        written_pbar.update(1)

    processed_pbar.close()
    written_pbar.close()

    read_executor.shutdown(wait=False)
    write_executor.shutdown(wait=False)
