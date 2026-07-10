from lib.settings.test_environment import EnvSettings_ITP

def local_test_env_settings():
    settings = EnvSettings_ITP()

    # Set your local paths here.

    settings.workspace_dir = '/'
    settings.save_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj/output'
    settings.results_path = '/home/yanjiezhang/Downloads/Dissertation/MainProj/output/test/tracking_results'    # Where to store tracking results
    settings.segmentation_path = '/home/yanjiezhang/Downloads/Dissertation/MainProj/output/test/segmentation_results'
    settings.network_path = '/home/yanjiezhang/Downloads/Dissertation/MainProj/output/test/networks'    # Where tracking networks are stored.
    settings.result_plot_path = '/home/yanjiezhang/Downloads/Dissertation/MainProj/output/test/result_plots'
    settings.fe108_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj/output/FE108/test'
    settings.visevent_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj/output/VisEvent/test'

    return settings

