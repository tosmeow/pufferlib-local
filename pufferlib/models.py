import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, encoder, decoder, network):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.network = network

    def initial_state(self, batch_size, device):
        return self.network.initial_state(batch_size, device)

    def forward_eval(self, x, state):
        h = self.encoder(x)
        h, state = self.network.forward_eval(h, state)
        logits, values = self.decoder(h)
        return logits, values, state

    def forward(self, x):
        B, TT = x.shape[:2]
        h = self.encoder(x.reshape(B*TT, *x.shape[2:]))
        h = self.network.forward_train(h.reshape(B, TT, -1))
        logits, values = self.decoder(h.reshape(B*TT, -1))
        return logits, values.reshape(B, TT)

class DefaultEncoder(nn.Module):
    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.encoder = nn.Linear(obs_size, hidden_size)

    def forward(self, observations):
        return self.encoder(observations.view(observations.shape[0], -1).float())

class LaggedBreakoutEncoder(DefaultEncoder):
    """Condition legacy per-frame velocity observations in pixels/second units.

    Converting a DefaultEncoder checkpoint requires dividing weight columns 4:6
    by 60. This preserves its forward function while improving input scaling.
    """
    def __init__(self, obs_size, hidden_size=128):
        if obs_size != 190:
            raise ValueError('LaggedBreakoutEncoder requires 190 observations')
        super().__init__(obs_size, hidden_size)
        scale = torch.ones(obs_size)
        scale[4:6] = 60.0
        self.register_buffer('_input_scale', scale, persistent=False)

    def forward(self, observations):
        flat = observations.reshape(observations.shape[0], -1).float()
        return self.encoder(flat * self._input_scale)

class _QueueReactiveCurriculumEncoder(nn.Module):
    """Checkpoint-stable encoder for the queue-reactive training curriculum."""
    OBS_SIZE = 44
    CORE_OBSERVATION_INDICES = tuple(range(16)) + (18, 21, 25)
    EXTRA_OBSERVATION_INDICES = (16, 17, 19, 20, 22, 23, 24) + tuple(range(26, 44))

    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        if obs_size != self.OBS_SIZE:
            raise ValueError(
                'Queue-reactive curriculum encoders require the 44-value '
                f'observation ABI, received {obs_size}'
            )

        self.core = nn.Linear(len(self.CORE_OBSERVATION_INDICES), hidden_size)
        self.extra = nn.Sequential(
            nn.Linear(len(self.EXTRA_OBSERVATION_INDICES), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        # The expanded encoder initially reproduces the core encoder exactly.
        # Training can then grow the residual branch without a policy jump.
        nn.init.zeros_(self.extra[-1].weight)

        self.register_buffer(
            '_core_indices',
            torch.tensor(self.CORE_OBSERVATION_INDICES, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            '_extra_indices',
            torch.tensor(self.EXTRA_OBSERVATION_INDICES, dtype=torch.long),
            persistent=False,
        )

    def _features(self, observations):
        flat = observations.reshape(observations.shape[0], -1).float()
        if flat.shape[1] != self.OBS_SIZE:
            raise ValueError(
                'Queue-reactive curriculum encoders require flattened '
                f'observations of size {self.OBS_SIZE}, received {flat.shape[1]}'
            )
        core = flat.index_select(1, self._core_indices)
        extra = flat.index_select(1, self._extra_indices)
        return core, extra

class QueueReactiveCoreEncoder(_QueueReactiveCurriculumEncoder):
    """Phase-one encoder: book, inventory, alpha, and time remaining only."""
    def __init__(self, obs_size, hidden_size=128):
        super().__init__(obs_size, hidden_size)
        self.extra.requires_grad_(False)

    def forward(self, observations):
        core, _ = self._features(observations)
        return self.core(core)

class QueueReactiveExpandedEncoder(_QueueReactiveCurriculumEncoder):
    """Later-phase encoder that adds all remaining observations as a residual."""
    def forward(self, observations):
        core, extra = self._features(observations)
        return self.core(core) + self.extra(extra)

class MinimalEntityEncoder(nn.Module):
    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.self_obs_size = 2
        self.point_obs_size = 4
        self.num_points = 16
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(self.self_obs_size + self.point_obs_size, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size),
        )

    def forward(self, observations):
        self_obs = observations[:, :self.self_obs_size].unsqueeze(1).expand(
            observations.shape[0], self.num_points, self.self_obs_size)
        point_obs = observations[:, self.self_obs_size:].reshape(
            observations.shape[0], self.num_points, self.point_obs_size)
        cat = torch.cat([self_obs, point_obs], dim=-1)
        return self.encoder(cat).max(dim=1)[0]

class DefaultDecoder(nn.Module):
    def __init__(self, nvec, hidden_size=128):
        super().__init__()
        self.nvec = tuple(nvec)
        self.is_continuous = sum(nvec) == len(nvec)

        if self.is_continuous:
            num_atns = len(nvec)
            self.decoder_mean = nn.Linear(hidden_size, num_atns)
            self.decoder_logstd = nn.Parameter(torch.zeros(1, num_atns))
        else:
            self.decoder = nn.Linear(hidden_size, int(np.sum(nvec)))

        self.value_function = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        if self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            logits = torch.distributions.Normal(mean, torch.exp(logstd))
        else:
            logits = self.decoder(hidden)
            if len(self.nvec) > 1:
                logits = logits.split(self.nvec, dim=1)

        values = self.value_function(hidden)
        return logits, values

class QueueReactiveMarketDistribution:
    """Conditional Noop/Market action heads consumed by ``sample_logits``."""
    def __init__(self, logits):
        self.logits = tuple(logits)

class QueueReactiveMarketDecoder(DefaultDecoder):
    """Phase-one decoder exposing Noop or Market without changing the ABI."""
    ACTION_SIZES = (5, 2, 8, 6)

    def __init__(self, nvec, hidden_size=128):
        if tuple(nvec) != self.ACTION_SIZES:
            raise ValueError(
                'QueueReactiveMarketDecoder requires MultiDiscrete'
                f'{self.ACTION_SIZES}, received {tuple(nvec)}'
            )
        super().__init__(nvec, hidden_size)

        type_mask = torch.tensor([True, False, False, True, False])
        depth_mask = torch.tensor([True, True, True, True, False, False, False, False])
        self.register_buffer('_type_mask', type_mask, persistent=False)
        self.register_buffer('_depth_mask', depth_mask, persistent=False)

        # Keep later action types/depths neutral while masked. Because the
        # parameter layout is identical to DefaultDecoder, its checkpoint can
        # subsequently be loaded strictly into the full action decoder.
        disabled_rows = (1, 2, 4, 11, 12, 13, 14)
        with torch.no_grad():
            rows = torch.tensor(disabled_rows, dtype=torch.long)
            self.decoder.weight.index_fill_(0, rows, 0)
            self.decoder.bias.index_fill_(0, rows, 0)

    def forward(self, hidden):
        logits, values = super().forward(hidden)
        type_logits, side_logits, depth_logits, size_logits = logits
        type_logits = type_logits.masked_fill(~self._type_mask, -torch.inf)
        depth_logits = depth_logits.masked_fill(~self._depth_mask, -torch.inf)
        return QueueReactiveMarketDistribution((
            type_logits,
            side_logits,
            depth_logits,
            size_logits,
        )), values

class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def initial_state(self, batch_size, device):
        return ()

    def forward_eval(self, h, state):
        return self.net(h), state

    def forward_train(self, h):
        return self.net(h)

class MinGRU(nn.Module):
    # https://arxiv.org/abs/2410.01201v1
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)
        ])

    def _g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def _log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def _highway(self, x, out, proj):
        g = proj.sigmoid()
        return g * out + (1.0 - g) * x

    def _heinsen_scan(self, log_coeffs, log_values):
        a_star = log_coeffs.cumsum(dim=1)
        return (a_star + (log_values - a_star).logcumsumexp(dim=1)).exp()

    def initial_state(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),)

    def forward_eval(self, h, state):
        state = state[0]
        assert state.shape[1] == h.shape[0]
        h = h.unsqueeze(1)
        state_out = []
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            out = torch.lerp(state[i:i+1].transpose(0, 1), self._g(hidden), gate.sigmoid())
            h = self._highway(h, out, proj)
            state_out.append(out[:, -1:])
        return h.squeeze(1), (torch.stack(state_out, 0).squeeze(2),)

    def forward_train(self, h):
        T = h.shape[1]
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            log_coeffs = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self._log_g(hidden)
            out = self._heinsen_scan(log_coeffs, log_values)[:, -T:]
            h = self._highway(h, out, proj)
        return h

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.lstm, f'weight_ih_l{i}')
            w_hh = getattr(self.lstm, f'weight_hh_l{i}')
            b_ih = getattr(self.lstm, f'bias_ih_l{i}')
            b_hh = getattr(self.lstm, f'bias_hh_l{i}')
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, h, state):
        assert state[0].shape[1] == state[1].shape[1] == h.shape[0]
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c
        return h, (lstm_h, lstm_c)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h, _ = self.lstm(h)
        return h.transpose(0, 1)

class GRU(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norm = torch.nn.RMSNorm(hidden_size)

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.gru, f'weight_ih_l{i}')
            w_hh = getattr(self.gru, f'weight_hh_l{i}')
            b_ih = getattr(self.gru, f'bias_ih_l{i}')
            b_hh = getattr(self.gru, f'bias_hh_l{i}')
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h,)

    def forward_eval(self, h, state):
        assert state[0].shape[1] == h.shape[0]
        state = state[0]
        for i in range(self.num_layers):
            h_in = h
            h = self.cell[i](h, state[i])
            state[i] = h
            h = h + h_in
            h = self.norm(h)
        return h, (state,)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h_in = h
        h, _ = self.gru(h)
        h = h + h_in
        h = self.norm(h)
        return h.transpose(0, 1)

class NatureEncoder(nn.Module):
    '''NatureCNN encoder (Mnih et al. 2015). Returns [batch, hidden_size].'''
    def __init__(self, env, hidden_size=512, framestack=1, flat_size=64*7*7,
            channels_last=False, downsample=1, **kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.network = nn.Sequential(
            nn.Conv2d(framestack, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, 3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class ImpalaEncoder(nn.Module):
    '''IMPALA ResNet encoder (Espeholt et al. 2018). Returns [batch, hidden_size].'''
    def __init__(self, env, hidden_size=256, cnn_width=16, **kwargs):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(shape[0] * shape[1] * shape[2], hidden_size),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, observations):
        return self.network(observations.permute(0, 3, 1, 2).float() / 255.0)
