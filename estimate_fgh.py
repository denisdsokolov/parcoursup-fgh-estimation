import pandas as pd
import numpy as np
from collections import OrderedDict

import collections, collections.abc
if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable

for _name in ("Mapping", "MutableMapping", "Sequence", "Set", "MutableSet"):
        if not hasattr(collections, _name):
                setattr(collections, _name, getattr(collections.abc, _name))

import pylogit
from pylogit import create_choice_model

def estimate(panel):
    """Return fitted pylogit model (with ASC_diffs)"""
    spec = OrderedDict([
            ("near_is", "all_same"),   # beta1
            #("near_sch", "all_same"),   # beta2
            ("bOnly_sch", "all_same"),   # beta3
            #("br_sch", "all_same"),   # beta4
            #("br_near", "all_same"),   # beta5
            #("near_nat", "all_same"),   # beta6
            #("near_sex", "all_same"),   # beta7
            #("bShare_sex", "all_same"),   # beta8
            #("rShare_sex", "all_same"),   # beta9
            #("rShare_sch", "all_same"),   # beta10
            #("bShare_near", "all_same"),   # beta11
            #("bShare_sch", "all_same"),   # beta12
            #("rShare_near", "all_same"),   # beta13
            ("bOnly_bOnlyWish", "all_same"),   # beta14
            ("br_brWishRes", "all_same"),   # beta15
            ("logQuota_nat", "all_same"),   # beta16
            ("logQuota_near", "all_same"),   # beta17
            ("logQuota_sex", "all_same"),   # beta16
            ("logQuota_sch", "all_same"),   # beta17
            ("out_sch", [0]),   # beta18 (alt_id 0 only)
            ("out_sex", [0]),   # beta19 (alt_id 0 only)
            ("out_nat", [0]),   # beta20 (alt_id 0 only)
            ("out_wish", [0]),   # beta21 (alt_id 0 only)
            ("ASC", "all_diff"),   # alpha_s (sampled programmes)
    ])
    model = create_choice_model(panel,
                                alt_id_col="program_id",
                                obs_id_col="student_id",
                                choice_col="chosen",
                                specification=spec,
                                model_type="MNL")
    model.fit_mle(init_vals=np.zeros(model.design.shape[1]),
                  ridge=1e-6, print_res=True)
    return model