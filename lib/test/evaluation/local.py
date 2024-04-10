from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.davis_dir = ''
    settings.save_dir = '/home/ldd/Desktop/Projects/UAVmatch/workspace'
    settings.prj_dir = '/home/ldd/Desktop/Projects/UAVmatch'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    return settings

