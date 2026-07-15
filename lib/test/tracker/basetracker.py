import time
import torch
from lib.vis.visdom_cus import Visdom



class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, settings):
        self.settings = settings
        self.cfg = settings.cfg
        self.visdom = None

    def initialize(self, image, info: dict):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image, gt_bbox: torch.Tensor):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def visdom_draw_tracking(self, image, box, debug_level=1):
        if isinstance(box, dict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        self.visdom.register((image, *box), 'Tracking', debug_level, 'Tracking')

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        self.next_seq = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

            elif data['key'] == 'n':
                self.next_seq = True
