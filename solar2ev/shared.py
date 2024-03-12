#!/usr/bin/env python3

####################
# variable shared across modules, 
# but which are not really configuration data (either "user level" or "developer level")
# or defined at run time
####################


from enum import Enum
# key in result_dict, returned by inference from web
# use enum as keys, vs just hardcoded str, because I can

# but then need also to manage str as used as field in metric json file
#   json read in inference.update_model_tab() (to update gauge)
#   json written in train_and_assess.metric_to_json() (result from .evaluate)

# a lot of pain, not to "hardcode" "cat" and "reg" as keys in dict and json file

class model_type(Enum): # better than using str
    #cat = auto()
    cat = "cat"  # used as key when storing metrics to json
    reg = "reg"


feature_input_columns = None