import pdb

import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpps.models.decoders.base.decoder import Decoder
from tpps.utils.events import Events
from tpps.utils.nnplus import non_neg_param
from tpps.processes.hawkes_fast import decoder_fast as hawkes_decoder
# from tpp.processes.hawkes_slow import decoder_slow as hawkes_decoder


class HawkesDecoder(Decoder):
    """A parametric Hawkes Process decoder.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(self, marks: Optional[int] = 1, **kwargs):
        super(HawkesDecoder, self).__init__(name="hawkes", marks=marks)
        self.alpha = nn.Parameter(th.Tensor(self.marks, self.marks))
        self.beta = nn.Parameter(th.Tensor(self.marks, self.marks))
        self.mu = nn.Parameter(th.Tensor(self.marks))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.alpha)
        nn.init.uniform_(self.beta)
        nn.init.uniform_(self.mu)

    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.LongTensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None,
            time_prediction=False,
            sampling=False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        self.alpha.data = non_neg_param(self.alpha.data) #Returns only non-negative paramaters.
        self.mu.data = non_neg_param(self.mu.data)
        return hawkes_decoder(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs, 
            pos_delta_mask=pos_delta_mask, 
            is_event=is_event,
            alpha=self.alpha,
            beta=self.beta,
            mu=self.mu,
            marks=self.marks,
            time_prediction=time_prediction,
            sampling=sampling)
