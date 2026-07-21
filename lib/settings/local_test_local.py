from settings.test_environment import EnvSettings_ITP

def local_test_env_settings():
    settings = EnvSettings_ITP()

    # Set your local paths here.

    settings.workspace_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj'
    settings.save_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj/tracking_results'
    settings.results_txt_path = '/home/yanjiezhang/Downloads/Dissertation/MainProj/tracking_results/txt_results'
    settings.result_plot_path = '/home/yanjiezhang/Downloads/Dissertation/MainProj/tracking_results/plot_results'
    settings.checkpoint_path = ''
    settings.fe108_dir = '/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_GTP_lmdb'
    settings.visevent_dir = '/home/yanjiezhang/Downloads/Dissertation/dataset/VisEvent/test'

    return settings

