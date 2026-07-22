import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from aggregation.utils.general import clip_events, get_offset, unique_dir


def _generate_ch3(pos, last_pos, neg, last_neg, hidden, ch3_strength):
    pos_map = (last_pos == 0) & (pos != 0)
    neg_map = (last_neg == 0) & (neg != 0)
    hidden[pos_map] += ch3_strength
    hidden[neg_map] += ch3_strength
    return np.clip(hidden, 0, 255).astype(np.uint8)


def _generate_chs(
        events,
        last_pos,
        last_neg,
        last_hidden,
        width,
        height,
        ch12_strength,
        ch3_strength,
        ch12_decay_rate,
        ch3_decay_rate,
        first_frame=False,
):
    pos = last_pos.astype(np.float32) * ch12_decay_rate
    neg = last_neg.astype(np.float32) * ch12_decay_rate
    hidden = last_hidden.astype(np.float32) * ch3_decay_rate

    if len(events) > 0:
        mask_pos = events["polarity"] == 1
        mask_neg = ~mask_pos
        np.add.at(pos, (events["y"][mask_pos], events["x"][mask_pos]), ch12_strength)
        np.add.at(neg, (events["y"][mask_neg], events["x"][mask_neg]), ch12_strength)

    pos = np.clip(pos, 0, 255).astype(np.uint8)
    neg = np.clip(neg, 0, 255).astype(np.uint8)

    if first_frame:
        return pos, neg, hidden.astype(np.uint8)

    hidden = _generate_ch3(
        pos=pos,
        last_pos=last_pos,
        neg=neg,
        last_neg=last_neg,
        hidden=hidden,
        ch3_strength=ch3_strength,
    )
    return pos, neg, hidden


def _concat_channels(pos, neg, hidden):
    assert pos.shape == neg.shape == hidden.shape
    height, width = pos.shape
    gtp = np.zeros((3, height, width), dtype=np.uint8)
    gtp[0] = pos
    gtp[1] = neg
    gtp[2] = hidden
    return gtp


def _make_color_gtp(gtp):
    assert gtp.ndim == 3 and gtp.shape[0] == 3
    return gtp.transpose(1, 2, 0)


def _split_frame_events(frame_events, sub_frame_start):
    bins = []
    start = 0
    for sub_idx in range(len(sub_frame_start) - 1):
        if sub_idx == len(sub_frame_start) - 2:
            end = len(frame_events)
        else:
            boundary = sub_frame_start[sub_idx + 1]
            end = start
            while end < len(frame_events) and frame_events[end]["timestamp"] < boundary:
                end += 1
        bins.append(frame_events[start:end])
        start = end
    return bins


def _aggregate_frame_bins(
        frame_events,
        sub_frame_start,
        width,
        height,
        ch12_strength,
        ch3_strength,
        ch12_decay_rate,
        ch3_decay_rate,
        agg_decay_rate,
        carry_hidden,
        first_frame,
):
    frame_bins = _split_frame_events(frame_events, sub_frame_start)

    pos_state = np.zeros((height, width), dtype=np.uint8)
    neg_state = np.zeros((height, width), dtype=np.uint8)
    hidden_state = carry_hidden.astype(np.uint8)

    pos_acc = np.zeros((height, width), dtype=np.float32)
    neg_acc = np.zeros((height, width), dtype=np.float32)
    hidden_acc = np.zeros((height, width), dtype=np.float32)

    per_bin_states = []
    num_bins = len(frame_bins)

    for bin_idx, sub_events in enumerate(frame_bins):
        pos_state, neg_state, hidden_state = _generate_chs(
            events=sub_events,
            last_pos=pos_state,
            last_neg=neg_state,
            last_hidden=hidden_state,
            width=width,
            height=height,
            ch12_strength=ch12_strength,
            ch3_strength=ch3_strength,
            ch12_decay_rate=ch12_decay_rate,
            ch3_decay_rate=ch3_decay_rate,
            first_frame=first_frame,
        )
        first_frame = False

        per_bin_states.append((
            pos_state.copy(),
            neg_state.copy(),
            hidden_state.copy(),
        ))

        bin_weight = agg_decay_rate ** (num_bins - 1 - bin_idx)
        pos_acc += bin_weight * pos_state.astype(np.float32)
        neg_acc += bin_weight * neg_state.astype(np.float32)
        hidden_acc += bin_weight * hidden_state.astype(np.float32)

    pos_out = np.clip(pos_acc, 0, 255).astype(np.uint8)
    neg_out = np.clip(neg_acc, 0, 255).astype(np.uint8)
    hidden_out = np.clip(hidden_acc, 0, 255).astype(np.uint8)

    return pos_out, neg_out, hidden_out, hidden_state, first_frame, per_bin_states


def _save_subbins(per_bin_states, img_dir, frame_idx):
    for sub_idx, (pos, neg, hidden) in enumerate(per_bin_states, start=1):
        gtp = _concat_channels(pos, neg, hidden)
        img = _make_color_gtp(gtp)
        cv2.imwrite(str(img_dir / f"frame_{frame_idx:06d}_sub_{sub_idx:02d}.png"), img)


def aedat4_to_FCTP(
        aedat_path,
        offset_path,
        out_dir,
        folder_name="imgs_FCTP",
        width=346,
        height=260,
        ch12_strength=40,
        ch3_strength=30,
        agg_decay_rate=0.7,
        ch12_decay_rate=0.8,
        ch3_decay_rate=0.9,
        num_bins=4,
        output_mode="per_bin",  # "aggregate", "per_bin", "both"
):
    aedat_path = Path(aedat_path)
    offset = get_offset(offset_path)

    events, frame_start_list = clip_events(aedat_path, offset)
    frame_start_num = len(frame_start_list)

    frame_idx = 1
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = unique_dir(output_dir / folder_name)
    img_dir.mkdir(exist_ok=True, parents=True)

    start_idx = 0
    last_hidden = np.zeros((height, width), dtype=np.uint8)
    first_frame = True

    for idx, event in enumerate(tqdm(events, leave=False, desc="Converting aedat4 to FCTP")):
        if frame_idx < frame_start_num and event["timestamp"] >= frame_start_list[frame_idx]:
            frame_events = events[start_idx:idx]
            sub_frame_start = np.linspace(
                frame_start_list[frame_idx - 1],
                frame_start_list[frame_idx],
                num_bins + 1,
            )

            pos, neg, hidden, last_hidden, first_frame, per_bin_states = _aggregate_frame_bins(
                frame_events=frame_events,
                sub_frame_start=sub_frame_start,
                width=width,
                height=height,
                ch12_strength=ch12_strength,
                ch3_strength=ch3_strength,
                ch12_decay_rate=ch12_decay_rate,
                ch3_decay_rate=ch3_decay_rate,
                agg_decay_rate=agg_decay_rate,
                carry_hidden=last_hidden,
                first_frame=first_frame,
            )

            if output_mode in ("per_bin", "both"):
                _save_subbins(per_bin_states, img_dir, frame_idx)

            if output_mode in ("aggregate", "both"):
                gtp = _concat_channels(pos, neg, hidden)
                gtp_img = _make_color_gtp(gtp)
                cv2.imwrite(str(img_dir / f"frame_{frame_idx:06d}.png"), gtp_img)

            start_idx = idx
            frame_idx += 1

    if start_idx < len(events):
        frame_events = events[start_idx:]

        if frame_start_num >= 2:
            if frame_idx < frame_start_num:
                frame_start = frame_start_list[frame_idx - 1]
                frame_end = frame_start_list[frame_idx]
            else:
                frame_start = frame_start_list[-2]
                frame_end = frame_start_list[-1]
            sub_frame_start = np.linspace(frame_start, frame_end, num_bins + 1)
        else:
            first_ts = frame_events["timestamp"][0]
            last_ts = frame_events["timestamp"][-1]
            sub_frame_start = np.linspace(first_ts, last_ts, num_bins + 1)

        pos, neg, hidden, _, _, per_bin_states = _aggregate_frame_bins(
            frame_events=frame_events,
            sub_frame_start=sub_frame_start,
            width=width,
            height=height,
            ch12_strength=ch12_strength,
            ch3_strength=ch3_strength,
            ch12_decay_rate=ch12_decay_rate,
            ch3_decay_rate=ch3_decay_rate,
            agg_decay_rate=agg_decay_rate,
            carry_hidden=last_hidden,
            first_frame=first_frame,
        )

        if output_mode in ("per_bin", "both"):
            _save_subbins(per_bin_states, img_dir, frame_idx)

        if output_mode in ("aggregate", "both"):
            gtp = _concat_channels(pos, neg, hidden)
            gtp_img = _make_color_gtp(gtp)
            cv2.imwrite(str(img_dir / f"frame_{frame_idx:06d}.png"), gtp_img)

    ts_df = pd.DataFrame(
        {
            "frame_idx": range(1, len(frame_start_list)),
            "timestamp": frame_start_list[:-1],
        }
    )
    ts_df.to_csv(output_dir / f"{folder_name}_timestamp.csv", index=False)


if __name__ == "__main__":
    aedat_path = "/media/yanjiezhang/ian/dataset/FE108_raw/train/dove/events.aedat4"
    out_dir = "/media/yanjiezhang/ian/dataset/ship/dove/"
    width, height = 346, 260
    offset = "/media/yanjiezhang/ian/dataset/FE108_raw/offset.txt"

    aedat4_to_FCTP(
        Path(aedat_path),
        Path(offset),
        out_dir,
        width=width,
        height=height,
        num_bins=4,
        ch12_strength=40,
        ch3_strength=30,
        agg_decay_rate=0.1,
        ch12_decay_rate=0,
        ch3_decay_rate=0.7,
        output_mode="aggregate",
    )
