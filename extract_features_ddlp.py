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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    duplicate_on_episode_start = config.get("duplicate_on_episode_start", False) # populate the context with duplicated frame in the beginning of the episode

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
    dynamics = config.get('dynamics', True)
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
                              max_beta_coef=max_beta_coef, action_dim=action_dim, mu_scale_prior=mu_scale_prior,
                              dynamics=dynamics)

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


def get_fg_representation(dlp_output):
    rep = torch.cat((dlp_output['z'], dlp_output['mu_scale'], dlp_output['mu_depth'], dlp_output['mu_features'],
                     dlp_output['obj_on'].unsqueeze(dim=-1)), dim=-1).detach().cpu().numpy()
    return rep


def get_bg_representation(dlp_output):
    rep = dlp_output['z_bg'].detach().cpu().numpy()
    return rep


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


def write_episode_data(dataset_path, episode_id, representations, actions, rewards):
    # actions -> episode_length, horizon, action_dim
    # rewards -> episode_length, 1
    with h5py.File(dataset_path, 'a') as hf:
        grp = hf.create_group(str(episode_id))
        grp.create_dataset('actions', data=actions)
        grp.create_dataset('rewards', data=rewards)
        for key, value in representations.items():
            grp.create_dataset(key, data=value)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
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
    parser.add_argument('--use_autoregression', type=str2bool, default=False)
    parser.add_argument('--action_history', type=str2bool, default=False)
    parser.add_argument('--prediction_horizon', type=int, default=0)
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
        timestep_horizon = model.timestep_horizon
        data = read_futures.popleft().result()
        x = data['images'].to(args.device)
        representations = collections.defaultdict(list)
        if args.use_autoregression:
            start_image = x[:1].expand(timestep_horizon + 1, -1, -1, -1).unsqueeze(0)
            dlp_output = model(start_image, deterministic=True, x_prior=start_image, warmup=False, noisy=False, predict_next=False,
                                   sequential=True, train_enc_prior=config['train_enc_prior'],
                                   num_static_frames=config['num_static_frames'])
            z_prev = dlp_output['z'][-1:]
            z_scale_prev = dlp_output['z_scale'][-1:]
            cropped_objects_prev = dlp_output['cropped_objects_original'][-1:]
            for key in ('z', 'mu_scale', 'mu_depth', 'mu_features', 'obj_on', 'z_bg'):
                representations[key].append(dlp_output[key][-1].detach().cpu().numpy())

            for step_image in x[1:]:
                step_image = step_image.unsqueeze(0)
                fg_dict = model.fg_module.encode_all(step_image, deterministic=False, warmup=False, noisy=False, kp_init=z_prev,
                                                     cropped_objects_prev=cropped_objects_prev.flatten(end_dim=1),
                                                     scale_prev=z_scale_prev, refinement_iter=False)
                for key in ('z', 'mu_scale', 'mu_depth', 'mu_features', 'obj_on'):
                    representations[key].append(dlp_output[key][0].detach().cpu().numpy())

                bg_enc_mask = model.get_bg_mask_from_particle_glimpses(fg_dict['z'], fg_dict['obj_on'], mask_size=step_image.shape[-1])
                bg_dict = model.bg_module(step_image, bg_enc_mask, deterministic=False)
                representations['z_bg'].append(bg_dict['z_bg'][0].detach().cpu().numpy())

                z_prev = fg_dict['z']
                cropped_objects_prev = fg_dict['cropped_objects']
                z_scale_prev = fg_dict['z_scale']

            representations = {k: np.stack(v) for k, v in representations.items()}
            representations = {k: v.reshape((*v.shape[:2], -1)) for k, v in representations.items()}
        elif args.prediction_horizon > 0:
            actions = torch.as_tensor(data['actions'], device=args.device)
            actions = torch.cat(
                [torch.zeros(model.timestep_horizon - 1, *actions.size()[1:], dtype=actions.dtype, device=actions.device), actions],
                dim=0)
            actions = actions.unfold(dimension=0, size=model.timestep_horizon + args.prediction_horizon - 1, step=1).permute(
                (0, 2, 1)).contiguous()
            x = torch.cat([x[:1].expand((model.timestep_horizon - 1, -1, -1, -1)), x], dim=0)
            x = x.unfold(dimension=0, size=model.timestep_horizon, step=1).permute((0, 4, 1, 2, 3))[:actions.size()[0]].contiguous()
            rewards = torch.as_tensor(data['rewards'])
            rewards = rewards.unfold(dimension=0, size=args.prediction_horizon, step=1)
            for batch_indices in torch.split(torch.arange(x.size()[0], device=x.device), args.batch_size):
                batch_x = torch.index_select(x, dim=0, index=batch_indices)
                batch_actions = torch.index_select(actions, dim=0, index=batch_indices)
                dlp_output = model(batch_x, deterministic=True, x_prior=batch_x, warmup=False, noisy=False, predict_next=False,
                                   sequential=True, train_enc_prior=config['train_enc_prior'],
                                   num_static_frames=config['num_static_frames'])

                for key in ('z', 'z_scale', 'obj_on', 'z_depth', 'z_features', 'z_bg'):
                    shape = dlp_output[key].shape
                    dlp_output[key] = dlp_output[key].reshape(batch_indices.size()[0], model.timestep_horizon, *shape[1:])

                z, z_scale, z_obj_on, z_depth, z_features, z_bg_features = model.dyn_module.sample(
                    dlp_output['z'],
                    dlp_output['z_scale'],
                    dlp_output['obj_on'],
                    dlp_output['z_depth'],
                    dlp_output['z_features'],
                    dlp_output['z_bg'],
                    steps=args.prediction_horizon,
                    deterministic=True,
                    action=batch_actions)

                z = z.unfold(dimension=1, size=model.timestep_horizon, step=1).movedim(-1, 2)[:, 1:] # only predictions
                z_scale = z_scale.unfold(dimension=1, size=model.timestep_horizon, step=1).movedim(-1, 2)[:, 1:] # only predictions
                z_obj_on = z_obj_on.unfold(dimension=1, size=model.timestep_horizon, step=1).movedim(-1, 2)[:, 1:] # only predictions
                z_depth = z_depth.unfold(dimension=1, size=model.timestep_horizon, step=1).movedim(-1, 2)[:, 1:] # only predictions
                z_features = z_features.unfold(dimension=1, size=model.timestep_horizon, step=1).movedim(-1, 2)[:, 1:] # only predictions
                z_bg_features = z_bg_features.unfold(dimension=1, size=model.timestep_horizon, step=1).movedim(-1, 2)[:, 1:] # only predictions
                a = batch_actions.unfold(dimension=1, size=model.timestep_horizon, step=1).movedim(-1, 2)
                r = torch.index_select(rewards, dim=0, index=batch_indices.to(rewards.device))

                for key, feature in zip(('z', 'mu_scale', 'mu_depth', 'mu_features', 'obj_on', 'z_bg'), (z, z_scale, z_depth, z_features, z_obj_on, z_bg_features)):
                    value = feature.detach().cpu().numpy()
                    if key == 'obj_on':
                        value = np.expand_dims(value, axis=-1)
                    if key == 'z_bg':
                        value = np.expand_dims(value, axis=-2)

                    representations[key].append(value)

                representations['actions'].append(a.cpu().numpy())
                representations['rewards'].append(r.cpu().numpy())

            representations = {k: np.concatenate(v) for k, v in representations.items()}
        else:
            x = torch.cat([x[:1].expand((model.timestep_horizon - 1, -1, -1, -1)), x], dim=0)
            x = x.unfold(dimension=0, size=model.timestep_horizon, step=1).permute((0, 4, 1, 2, 3)).contiguous()
            for batch in torch.split(x, args.batch_size):
                dlp_output = model(batch, deterministic=True, x_prior=batch, warmup=False, noisy=False, predict_next=False,
                                   sequential=True, train_enc_prior=config['train_enc_prior'],
                                   num_static_frames=config['num_static_frames'])
                for key in ('z', 'mu_scale', 'mu_depth', 'mu_features', 'obj_on', 'z_bg'):
                    value = dlp_output[key].detach().cpu().numpy()
                    value = value.reshape((*batch.size()[:2], *value.shape[1:]))
                    if key == 'obj_on':
                        value = np.expand_dims(value, axis=-1)
                    if key == 'z_bg':
                        value = np.expand_dims(value, axis=-2)

                    representations[key].append(value)

            representations = {k: np.concatenate(v) for k, v in representations.items()}

        if 'actions' in representations:
            actions = representations.pop('actions')
        else:
            actions = data['actions']
            if args.action_history:
                actions = np.concatenate(
                    [np.zeros((model.timestep_horizon - 1, *actions.shape[1:]), dtype=np.float32,), actions],
                    axis=0
                )
                actions = sliding_window_view(actions, window_shape=model.timestep_horizon + args.prediction_horizon, axis=0).transpose((0, 2, 1))

        if 'rewards' in representations:
            rewards = representations.pop('rewards')
        else:
            rewards = data['rewards']

        write_futures.append(write_executor.submit(
            write_episode_data,
            os.path.join(args.target_dataset_path, f'{data["split"]}.hdf5'),
            data['episode_id'],
            representations,
            actions,
            rewards,
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
