import cv2, os
import matplotlib.pyplot as plt
import numpy as np


def vis_pred_detections_by_pair(t1_info_vis, t2_info_vis, save_path=''):
    t1_img_name, t2_img_name = t1_info_vis['img_meta']['img_name'], t2_info_vis['img_meta']['img_name']
    t1_img_ori, t1_tlbr_ori, t1_id_ori = t1_info_vis['img_meta']['image'], t1_info_vis['tlbr'], t1_info_vis['id']
    t2_img_ori, t2_tlbr_ori, t2_id_ori = t2_info_vis['img_meta']['image'], t2_info_vis['tlbr'], t2_info_vis['id']
    t2_score_mask = t2_info_vis['mask'].reshape(-1)
    t1_unmatched_idx, t2_unmatched_idx = t1_info_vis['unmatched'], t2_info_vis['unmatched']

    t1_img, t2_img = np.ascontiguousarray(t1_img_ori), np.ascontiguousarray(t2_img_ori)
    t1_tlbr, t2_tlbr = t1_tlbr_ori.cpu().numpy(), t2_tlbr_ori.cpu().numpy()
    t1_id, t2_id = t1_id_ori.reshape(-1).cpu().numpy(), t2_id_ori.reshape(-1).cpu().numpy()

    H, W, _ = t1_img.shape
    t1_tlbr = (t1_tlbr * 4).astype(int)
    t2_tlbr = (t2_tlbr * 4).astype(int)
    t1_tlbr = np.clip(t1_tlbr, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
    t2_tlbr = np.clip(t2_tlbr, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2

    ## visualize t1 with bbox and id
    t1_unmatched_num, t1_matched_num = 0, 0
    for idx, t1_box in enumerate(t1_tlbr):
        if idx in t1_unmatched_idx:
            font_color = (1, 0, 0)  # unmatched box id in RED
            t1_unmatched_num += 1
        else:
            font_color = (0, 1, 0)  # matched box id in green
            t1_matched_num += 1
        t1_img = cv2.rectangle(t1_img, (t1_box[0], t1_box[1]), (t1_box[2], t1_box[3]), (0, 1, 0), 2)
        cv2.putText(t1_img, str(t1_id[idx]), (t1_box[0], t1_box[1]), font_face, font_scale, font_color, font_thickness)

    ## visualize t2 with bbox and id
    t2_new_num, t2_rec_num, t2_matched_num = 0, 0, 0
    for idx, t2_box in enumerate(t2_tlbr):
        if t2_id[idx] != 0:
            box_thickness = 2
            if t2_score_mask[idx]:  # pred box with score higher than threshold
                if idx in t2_unmatched_idx:
                    box_color = (0, 0, 1)
                    font_color = (0, 0, 1)
                    t2_new_num += 1
                else:
                    box_color = (0, 1, 0)
                    font_color = (0, 1, 0)
                    t2_matched_num += 1
            else:  # recovered detection which score lower than threshold
                box_color = (1, 0, 0)
                font_color = (1, 0, 0)
                t2_rec_num += 1
        else:  # low score unmatched boxes
            box_thickness = 1
            box_color = (1, 1, 1)
            # box_color = (0.5, 0.5, 0.5)

        t2_img = cv2.rectangle(t2_img, (t2_box[0], t2_box[1]), (t2_box[2], t2_box[3]), box_color, box_thickness)
        if t2_id[idx] != 0:
            cv2.putText(t2_img, str(t2_id[idx]), (t2_box[0], t2_box[1]), font_face, font_scale, font_color,
                        font_thickness)

    plt.subplot(2, 1, 1)
    plt.text(-400, 50, 'T1 unmatched in red ID\nT2 recovered in red ID\nT2 new det in blue ID')

    plt.imshow(t1_img)
    plt.title(f'T1 - {t1_img_name}')
    plt.axis('off')
    plt.tight_layout(pad=0)

    plt.subplot(2, 1, 2)
    plt.text(-400, 50, f'T1 unmatched: {t1_unmatched_num}\nT1 matched: {t1_matched_num}\
                    \nT2 recovered: {t2_rec_num}\nT2 new: {t2_rec_num}\nT2 matched: {t2_matched_num}')
    plt.imshow(t2_img)
    plt.title(f'T2 - {t2_img_name}')
    plt.axis('off')
    plt.tight_layout(pad=0)

    if save_path != '':
        save_filename = os.path.join(save_path, t2_img_name + '.png')
        if os.path.exists(save_filename):
            os.remove(save_filename)
        plt.savefig(save_filename)
    else:
        plt.show()
    plt.clf()