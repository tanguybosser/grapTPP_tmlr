import torch as th
import torch.nn as nn

from typing import Optional, List

from tpps.models.decoders.log_normal_mixture import LogNormalMixtureDecoder

from tpps.utils.stability import epsilon

class LogNormalMixture_JD(LogNormalMixtureDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://arxiv.org/pdf/1909.12127.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            n_mixture: int,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            embedding_constraint: Optional[str] = None,
            emb_dim: Optional[int] = 2,
            mark_activation: Optional[str] = 'relu',
            name: Optional[str] = 'log-normal-mixture-jd',
            **kwargs):
        super(LogNormalMixture_JD, self).__init__(
            name=name,
            n_mixture=n_mixture,
            units_mlp=units_mlp,
            multi_labels=multi_labels,
            marks=marks,
            encoding=encoding,
            embedding_constraint=embedding_constraint, 
            emb_dim=emb_dim,
            mark_activation=mark_activation,
            **kwargs)
        
        self.mark_time = nn.Linear(
                in_features=self.encoding_size + self.input_size, out_features=units_mlp[1]
            )

    def log_mark_pmf(
            self, 
            query_representations:th.Tensor, 
            history_representations:th.Tensor):
        
        history_times = th.cat((history_representations, query_representations), dim=-1)
        p_m = th.softmax(
                self.marks2(
                    self.mark_activation(self.mark_time(history_times))), dim=-1)
        
        p_m = p_m + epsilon(dtype=p_m.dtype, device=p_m.device)

        return th.log(p_m)
    