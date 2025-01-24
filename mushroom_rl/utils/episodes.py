import torch
import numpy

def split_episodes(last, *arrays):
    """
    Split a array from shape (n_steps) to (n_episodes, max_episode_steps).
    """
    
    if last.sum().item() <= 1:
        return arrays if len(arrays) > 1 else arrays[0]

    row_idx, colum_idx, n_episodes, max_episode_steps = _torch_get_episode_idx(last) if type(last) == torch.Tensor else _numpy_get_episode_idx(last)
    episodes_arrays = []

    for array in arrays:
        if type(last) == torch.Tensor:
            array_ep = torch.zeros((n_episodes, max_episode_steps, *array.shape[1:]), dtype=array.dtype, device=array.device)
        else:
            array_ep = numpy.zeros((n_episodes, max_episode_steps, *array.shape[1:]), dtype=array.dtype)
        array_ep[row_idx, colum_idx] = array
        episodes_arrays.append(array_ep)

    return episodes_arrays if len(episodes_arrays) > 1 else episodes_arrays[0]

def unsplit_episodes(last, *episodes_arrays):
    """
    Unsplit a array from shape (n_episodes, max_episode_steps) to (n_steps).
    """
    
    if last.sum().item() <= 1:
        return episodes_arrays if len(episodes_arrays) > 1 else episodes_arrays[0]

    row_idx, colum_idx, _, _ = _torch_get_episode_idx(last) if type(last) == torch.Tensor else _numpy_get_episode_idx(last)

    arrays = []

    for episode_array in episodes_arrays:
        array = episode_array[row_idx, colum_idx]
        arrays.append(array)

    return arrays if len(arrays) > 1 else arrays[0]

def _torch_get_episode_idx(last):

    n_episodes = last.sum().item()
    last_idx = torch.nonzero(last).squeeze()
    episode_steps = torch.cat([torch.tensor([last_idx[0] + 1], device=last.device), last_idx[1:] - last_idx[:-1]])
    max_episode_steps = episode_steps.max().item()

    start_idx = torch.cat([torch.tensor([0], device=last.device), last_idx[:-1] + 1])
    row_idx = torch.arange(n_episodes, device=episode_steps.device).repeat_interleave(episode_steps)
    colum_idx = torch.arange(last.shape[0], device=last.device) - start_idx[row_idx]

    return row_idx, colum_idx, n_episodes, max_episode_steps

def _numpy_get_episode_idx(last):

    n_episodes = numpy.sum(last)
    last_idx = numpy.flatnonzero(last)
    episode_steps = numpy.concatenate(([last_idx[0] + 1], last_idx[1:] - last_idx[:-1]))
    max_episode_steps = numpy.max(episode_steps)

    start_idx = numpy.concatenate(([0], last_idx[:-1] + 1))
    row_idx = numpy.repeat(numpy.arange(n_episodes), episode_steps)
    column_idx = numpy.arange(last.shape[0]) - start_idx[row_idx]

    return row_idx, column_idx, n_episodes, max_episode_steps
