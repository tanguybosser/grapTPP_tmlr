import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.rmtpp import RMTPPDecoder
from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, subtract_exp, check_tensor

class RMTPP_JD(RMTPPDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            mark_activation: Optional[str] = 'relu',
            hist_time_grouping: Optional[str] = 'summation',
            name: Optional[str] = 'rmtpp-jd',
            **kwargs):
        super(RMTPP_JD, self).__init__(
            name=name,
            units_mlp=units_mlp,
            multi_labels=multi_labels,
            marks=marks,
            encoding=encoding,
            mark_activation=mark_activation,
            **kwargs)
        self.mark_time = nn.Linear(
                in_features=self.encoding_size + self.input_size, out_features=units_mlp[1])

        
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