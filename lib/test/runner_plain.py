import importlib
import os
import time
import numpy as np
from tqdm import tqdm

from lib.test.tracker.mistracker_plain import MISTracker


class MISTrackRunnerPlain:
    """wrapper the tracker for testing"""
    def __init__(self, dataset, settings):
        self.tracker = MISTracker(settings)
        self.dataset = dataset
        self.settings = settings
        self.cfg = settings.cfg
        self.save_dir = self.settings.env.results_txt_path
        os.makedirs(self.save_dir, exist_ok=True)


    def _save_pred_boxes(self, out_dict):
        pred_boxes = np.array(out_dict['pred_boxes'], dtype=int)
        save_file = os.path.join(self.save_dir, self.settings.cur_seq_name + '.txt')
        np.savetxt(save_file, pred_boxes, delimiter='\t', fmt='%d')

    def run_sequence(self, seq_id):
        '''
        data = {
            'search': torch.from_numpy(search_array).float(), # 0-255
            'search_anno': torch.from_numpy(search_anno_array).float()
        }'''
        self.settings.cur_seq_name = self.dataset._get_seq_name(seq_id)
        # inputs fed into the tracker should be (T,C,H,W)
        data = self.dataset[seq_id]
        search_anno = data['search_anno'] # (L,4)
        search_array = data['search'] # (L,T,C,H,W)
        gt_init_bbox = search_anno[0]
        first_frame = search_array[0]
        out_dict = {'pred_boxes': [np.round(gt_init_bbox.detach().cpu().numpy()).astype(int)]} # (L,4) (x,y,w,h)

        self.tracker.initialize(first_frame, gt_init_bbox)

        start_time = time.time()
        for frame_id in tqdm(range(1, search_array.shape[0]), desc=f'Running sequence {self.settings.cur_seq_name}'):
            cur_image = search_array[frame_id] # (T,C,H,W)
            cur_anno = search_anno[frame_id] # (4,)
            pred_dict = self.tracker.track(cur_image, cur_anno)
            out_dict['pred_boxes'].append(np.round(pred_dict['pred_bbox'].detach().cpu().numpy()).astype(int))
        end_time = time.time()

        if self.settings.cfg.TEST.DEBUG < 1:
            print(f'\033[91mFPS({self.settings.cur_seq_name}):\033[0m {search_array.shape[0] / (end_time - start_time):.2f}')
            self._save_pred_boxes(out_dict)
        return out_dict

    def run_sequence_full_image(self, seq_id):
        '''
        data = {
            'search': torch.from_numpy(search_array).float(), # 0-255
            'search_anno': torch.from_numpy(search_anno_array).float()
        }'''
        self.settings.cur_seq_name = self.dataset._get_seq_name(seq_id)
        # inputs fed into the tracker should be (T,C,H,W)
        data = self.dataset[seq_id]
        search_anno = data['search_anno'] # (L,4)
        search_array = data['search'] # (L,T,C,H,W)
        gt_init_bbox = search_anno[0]
        first_frame = search_array[0]
        out_dict = {'pred_boxes': [np.round(gt_init_bbox.detach().cpu().numpy()).astype(int)]} # (L,4) (x,y,w,h)

        self.tracker.initialize(first_frame, gt_init_bbox)

        total_time = 0
        for frame_id in tqdm(range(1, search_array.shape[0]), desc=f'Running sequence {self.settings.cur_seq_name}'):
            cur_image = search_array[frame_id] # (T,C,H,W)
            cur_anno = search_anno[frame_id] # (4,)
            start_time = time.time()
            pred_dict = self.tracker.track_full_image(cur_image, cur_anno)
            end_time = time.time()
            total_time += (end_time - start_time)

            out_dict['pred_boxes'].append(np.round(pred_dict['pred_bbox'].detach().cpu().numpy()).astype(int))

        if self.settings.cfg.TEST.DEBUG < 1:
            print(f'\033[93mFPS({self.settings.cur_seq_name}):\033[0m {search_array.shape[0] / total_time:.2f}')
            self._save_pred_boxes(out_dict)
        return out_dict

    def run_dataset(self):
        for seq_id in range(self.dataset._get_num_seqs()):
            self.run_sequence(seq_id)

    def run_dataset_full_image(self):
        for seq_id in range(self.dataset._get_num_seqs()):
            self.run_sequence_full_image(seq_id)


if __name__ == '__main__':
    from lib.settings.settings import Settings
    from lib.test.data.dataset import FE108Dataset
    from lib.config.loader import load_from_yaml
    settings = Settings(training=False)
    settings.cfg = load_from_yaml("/home/yanjiezhang/Downloads/Dissertation/MainProj/experiments/fe108_mistrack.yaml")

    dataset = FE108Dataset(settings.env.fe108_dir, split='test', T=1)
    runner = MISTrackRunnerPlain(dataset, settings)
    runner.run_dataset()
    # runner.run_sequence(2)




