import torch.utils.data
from lib.utils.lmdb_loader import decode_txt, decode_img, decode_json

class BaseSeqDataset(torch.utils.data.Dataset):
    """ Base class for video datasets """

    def __init__(self, root, split: str):
        """
        """
        self.lmdb = root
        self.image_loader = decode_img
        self.json_loader = decode_json
        self.txt_loader = decode_txt
        self.split = split
        self.meta = None


    def __len__(self):
        """
        Returns the number of videos in the dataset
        :return:
        """
        return self.get_num_seqs()

    def __getitem__(self, item: tuple):
        """
        Returns an entire sequence and ground truth or a clip of one sequence and ground truth
        items: (seq_id, frame_start_id, L, P)
                L is the length of the clip, P is the step of predictions
        """
        raise NotImplementedError

    def _get_seq_name(self, seq_id):
        """ Name of the sequence
        returns:
            string - Name of the sequence
        """
        raise NotImplementedError

    def _get_num_seqs(self):
        """ Number of sequences in a dataset"""
        raise NotImplementedError

    def _get_num_frames(self, seq_id):
        """
        Total number of frames in a single sequence
        """
        raise NotImplementedError

    def _get_fame_names(self, seq_id, frame_ids):
        """
        get name of the frames
        :return:
        """
        raise NotImplementedError

    def _get_frame_keys(self, seq_id, frame_ids):
        raise NotImplementedError

    def _get_frames(self, seq_id, frame_ids):
        raise NotImplementedError

    def _get_annos(self, seq_id, frame_ids):
        raise NotImplementedError
