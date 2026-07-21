from settings.test_environment import EnvSettings_ITP

def local_test_env_settings():
    settings = EnvSettings_ITP()

    # Set your local paths here.

    settings.workspace_dir = '/user/work/rm25043/Dissertation/MainProj'
    settings.save_dir = '/user/work/rm25043/Dissertation/MainProj/tracking_results'
    settings.results_txt_path = '/user/work/rm25043/Dissertation/MainProj/tracking_results/txt_results'
    settings.result_plot_path = '/user/work/rm25043/Dissertation/MainProj/tracking_results/plot_results'
    settings.checkpoint_path = ''
    settings.fe108_dir = '/user/work/rm25043/Dissertation/dataset/FE108_nbinsGTP_lmdb'
    settings.visevent_dir = '/user/work/rm25043/Dissertation/dataset/VisEvent/test'

    return settings

