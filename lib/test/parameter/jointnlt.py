from lib.config.jointnlt.config import update_config_from_file, cfg
from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings


def parameters(yaml_name: str, model: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    print(f'save_dir:{save_dir}')
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/jointnlt/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.grounding_size = cfg.TEST.GROUNDING_SIZE
    params.grounding_factor = cfg.TEST.GROUNDING_FACTOR
    # Network checkpoint path
    if model is None:
        raise NotImplementedError("Please set proper model to test.")
    else:
        params.checkpoint = os.path.join(save_dir, "checkpoints/%s" % model)
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
