import torch

from mushroom_rl.core.serialization import Serializable
from mushroom_rl.utils.torch import TorchUtils


class TorchDataset(Serializable):
    def __init__(self, state_type, state_shape, action_type, action_shape, reward_shape, flag_shape,
                 policy_state_shape, mask_shape, device=None):

        self._device = TorchUtils.get_device(device)
        self._state_type = state_type
        self._action_type = action_type

        self._states = torch.empty(*state_shape, dtype=self._state_type, device=self._device)
        self._actions = torch.empty(*action_shape, dtype=self._action_type, device=self._device)
        self._rewards = torch.empty(*reward_shape, dtype=torch.float, device=self._device)
        self._next_states = torch.empty(*state_shape, dtype=self._state_type, device=self._device)
        self._absorbing = torch.empty(flag_shape, dtype=torch.bool, device=self._device)
        self._last = torch.empty(flag_shape, dtype=torch.bool, device=self._device)
        self._len = 0

        if policy_state_shape is None:
            self._policy_states = None
            self._policy_next_states = None
        else:
            self._policy_states = torch.empty(policy_state_shape, dtype=torch.float, device=self._device)
            self._policy_next_states = torch.empty(policy_state_shape, dtype=torch.float, device=self._device)

        if mask_shape is None:
            self._mask = None
        else:
            self._mask = torch.empty(mask_shape, dtype=torch.bool, device=self._device)

        self._add_all_save_attr()

    @classmethod
    def create_new_instance(cls, dataset=None):
        """
        Creates an empty instance of the Dataset and populates essential data structures

        Args:
            dataset (NumpyDataset, None): a template dataset to be used to create the new instance.

        Returns:
            A new empty instance of the dataset.

        """
        new_dataset = cls.__new__(cls)

        if dataset is not None:
            new_dataset._device = dataset._device
            new_dataset._state_type = dataset._state_type
            new_dataset._action_type = dataset._action_type
        else:
            new_dataset._device = None
            new_dataset._state_type = None
            new_dataset._action_type = None

        new_dataset._states = None
        new_dataset._actions = None
        new_dataset._rewards = None
        new_dataset._next_states = None
        new_dataset._absorbing = None
        new_dataset._last = None
        new_dataset._len = None
        new_dataset._policy_states = None
        new_dataset._policy_next_states = None
        new_dataset._mask = None

        new_dataset._add_all_save_attr()

        return new_dataset

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts,
                   policy_states=None, policy_next_states=None):
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states)
            actions = torch.as_tensor(actions)
            rewards = torch.as_tensor(rewards)
            next_states = torch.as_tensor(next_states)
            absorbings = torch.as_tensor(absorbings)
            lasts = torch.as_tensor(lasts)

        dataset = cls.create_new_instance()

        dataset._state_type = states.dtype
        dataset._action_type = actions.dtype

        dataset._states = torch.as_tensor(states)
        dataset._actions = torch.as_tensor(actions)
        dataset._rewards = torch.as_tensor(rewards)
        dataset._next_states = torch.as_tensor(next_states)
        dataset._absorbing = torch.as_tensor(absorbings, dtype=torch.bool)
        dataset._last = torch.as_tensor(lasts, dtype=torch.bool)
        dataset._len = len(lasts)

        if policy_states is not None and policy_next_states is not None:
            if not isinstance(policy_states, torch.Tensor):
                policy_states = torch.as_tensor(policy_states)
                policy_next_states = torch.as_tensor(policy_next_states)

            dataset._policy_states = policy_states
            dataset._policy_next_states = policy_next_states
        else:
            dataset._policy_states = None
            dataset._policy_next_states = None

        return dataset

    def __len__(self):
        return self._len

    def append(self, state, action, reward, next_state, absorbing, last, policy_state=None, policy_next_state=None,
               mask=None):
        i = self._len   # todo: handle index out of bounds?

        if mask is None:
            self._states[i] = state
            self._actions[i] = action
            self._rewards[i] = reward
            self._next_states[i] = next_state
            self._absorbing[i] = absorbing
            self._last[i] = last

            if self.is_stateful:
                self._policy_states[i] = policy_state
                self._policy_next_states[i] = policy_next_state
        else:
            n_active_envs = self._states.shape[1]

            self._states[i] = state[:n_active_envs]
            self._actions[i] = action[:n_active_envs]
            self._rewards[i] = reward[:n_active_envs]
            self._next_states[i] = next_state[:n_active_envs]
            self._absorbing[i] = absorbing[:n_active_envs]
            self._last[i] = last[:n_active_envs]

            if self.is_stateful:
                self._policy_states[i] = policy_state[:n_active_envs]
                self._policy_next_states[i] = policy_next_state[:n_active_envs]

            self._mask[i] = mask[:n_active_envs]

        self._len += 1

    def clear(self):
        self._states = torch.empty_like(self._states)
        self._actions = torch.empty_like(self._actions)
        self._rewards = torch.empty_like(self._rewards)
        self._next_states = torch.empty_like(self._next_states)
        self._absorbing = torch.empty_like(self._absorbing)
        self._last = torch.empty_like(self._last)

        if self.is_stateful:
            self._policy_states = torch.empty_like(self._policy_states)
            self._policy_next_states = torch.empty_like(self._policy_next_states)

        self._len = 0

    def get_view(self, index, copy=False):
        view = self.create_new_instance(self)

        if copy:
            view._states = torch.empty_like(self._states)
            view._actions = torch.empty_like(self._actions)
            view._rewards = torch.empty_like(self._rewards)
            view._next_states = torch.empty_like(self._next_states)
            view._absorbing = torch.empty_like(self._absorbing)
            view._last = torch.empty_like(self._last)

            new_states = self.state[index, ...]
            new_len = new_states.shape[0]

            view._states[:new_len] = new_states
            view._actions[:new_len] = self.action[index, ...]
            view._rewards[:new_len] = self.reward[index, ...]
            view._next_states[:new_len] = self.next_state[index, ...]
            view._absorbing[:new_len] = self.absorbing[index, ...]
            view._last[:new_len] = self.last[index, ...]
            view._len = new_len

            if self.is_stateful:
                view._policy_states = torch.empty_like(self._policy_states)
                view._policy_next_states = torch.empty_like(self._policy_next_states)

                view._policy_states[:new_len] = self._policy_states[index, ...]
                view._policy_next_states[:new_len] = self._policy_next_states[index, ...]

            if self._mask is not None:
                view._mask = torch.empty_like(self._mask)
                view._mask[:new_len] = self._mask[index, ...]
        else:
            view._states = self.state[index, ...]
            view._actions = self.action[index, ...]
            view._rewards = self.reward[index, ...]
            view._next_states = self.next_state[index, ...]
            view._absorbing = self.absorbing[index, ...]
            view._last = self.last[index, ...]
            view._len = view._states.shape[0]

            if self.is_stateful:
                view._policy_states = self._policy_states[index, ...]
                view._policy_next_states = self._policy_next_states[index, ...]

            if self._mask is not None:
                view._mask = self._mask[index, ...]

        return view

    def __getitem__(self, index):
        return self._states[index], self._actions[index], self._rewards[index], self._next_states[index], \
               self._absorbing[index], self._last[index]

    def __add__(self, other):
        result = self.create_new_instance(self)

        result._states = torch.concatenate((self.state, other.state))
        result._actions = torch.concatenate((self.action, other.action))
        result._rewards = torch.concatenate((self.reward, other.reward))
        result._next_states = torch.concatenate((self.next_state, other.next_state))
        result._absorbing = torch.concatenate((self.absorbing, other.absorbing))
        result._last = torch.concatenate((self.last, other.last))
        result._last[len(self) - 1] = True
        result._len = len(self) + len(other)

        if self.is_stateful:
            result._policy_states = torch.concatenate((self.policy_state, other.policy_state))
            result._policy_next_states = torch.concatenate((self.policy_next_state, other.policy_next_state))

        return result

    @property
    def state(self):
        return self._states[:len(self)]

    @property
    def action(self):
        return self._actions[:len(self)]

    @property
    def reward(self):
        return self._rewards[:len(self)]

    @property
    def next_state(self):
        return self._next_states[:len(self)]

    @property
    def absorbing(self):
        return self._absorbing[:len(self)]

    @property
    def last(self):
        return self._last[:len(self)]

    @property
    def policy_state(self):
        return self._policy_states[:len(self)]

    @property
    def policy_next_state(self):
        return self._policy_next_states[:len(self)]

    @property
    def is_stateful(self):
        return self._policy_states is not None

    @property
    def mask(self):
        return self._mask[:len(self)]

    @mask.setter
    def mask(self, new_mask):
        self._mask[:len(self)] = new_mask

    @property
    def n_episodes(self):
        n_episodes = self.last.sum()

        if not self.last[-1]:
            n_episodes += 1

        return n_episodes

    def _add_all_save_attr(self):
        self._add_save_attr(
            _state_type='primitive',
            _action_type='primitive',
            _device='primitive',
            _states='torch',
            _actions='torch',
            _rewards='torch',
            _next_states='torch',
            _absorbing='torch',
            _last='torch',
            _policy_states='numpy',
            _policy_next_states='numpy',
            _len='primitive'
        )