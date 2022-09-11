"""
establish dataset. Each sample in shape [ct_5_pre + ct_5_mid + ct_5_nex + 5 x ct_1_mid + 5 x penalty, 512, 512]

ct_5 means ct slice with 5 mm thickness; ct_1 means ct slice with 1mm thickness

model input three consecutive ct_5, output up-sampled five ct_1 for the middle ct_5
there are 5 penalty weight array to guide the training, so the shape is [13, 512, 512] for each sample

About the penalty weight array:
We classify each rescaled_ct into six semantic: 1) surface of airway/blood vessel inside lung; 2) airway, blood vessel;
3) airway, blood vessel, lesion; 4) inside the lung or heart; outside the lung and heart
The penalty weight sum for each component is the same; each voxel for same component has the same penalty weight
The final penalty weight is the sum for all 5 penalties. Like for voxel on surface of airway, it will have weights
from semantic 1, 2, 3 and 4, while voxel outside lung and heart only benefit from semantic 5.

pre-requite: rescaled_ct, lung_mask, blood_mask, airway mask, lesion mask
"""
import numpy as np
import Tool_Functions.Functions as Functions
import analysis.get_surface_rim_adjacent_mean as get_surface
import os
from analysis.connectivity import select_region


def slice_one_rescaled_ct(rescaled_ct, lung, blood, airway, lesion, heart, interval=2):
    rescaled_ct[rescaled_ct > 1] = 1
    rescaled_ct[rescaled_ct < -0.25] = -0.25
    inside_lung = np.where(lung > 0)
    z_min = max(np.min(inside_lung[2]) - 20, 20)
    z_max = min(np.max(inside_lung[2]) + 20, 480)

    blood_airway_combined = np.array(airway + blood > 0.5, "float32")

    mask_semantic_1 = get_surface.get_surface(blood_airway_combined, outer=False, strict=False) * lung
    mask_semantic_2 = blood_airway_combined
    mask_semantic_3 = np.array(np.clip(blood_airway_combined + lesion, 0, 1))
    mask_semantic_4 = np.clip(lung + heart, 0, 1)
    mask_semantic_5 = 1 - mask_semantic_4

    semantic_weight_sum = 1000000

    final_penalty_mask = np.array(semantic_weight_sum / np.sum(mask_semantic_1) * mask_semantic_1, 'float32')
    final_penalty_mask = final_penalty_mask + semantic_weight_sum / np.sum(mask_semantic_2) * mask_semantic_2
    final_penalty_mask = final_penalty_mask + semantic_weight_sum / np.sum(mask_semantic_3) * mask_semantic_3
    final_penalty_mask = final_penalty_mask + semantic_weight_sum / np.sum(mask_semantic_4) * mask_semantic_4
    final_penalty_mask = final_penalty_mask + semantic_weight_sum / np.sum(mask_semantic_5) * mask_semantic_5

    def generate_one_sample(mid_z):
        training_sample = {}

        # the model inputs, i.e., three ct_5
        model_input = np.zeros([5, 512, 512])
        model_input[0, :, :] = np.average(rescaled_ct[:, :, (mid_z - 12): (mid_z - 7)], axis=2)  # ct_5_pre
        model_input[1, :, :] = np.average(rescaled_ct[:, :, (mid_z - 7): (mid_z - 2)], axis=2)  # ct_5_mid
        model_input[2, :, :] = np.average(rescaled_ct[:, :, (mid_z - 2): (mid_z + 3)], axis=2)  # ct_5_nex
        model_input[3, :, :] = np.average(rescaled_ct[:, :, (mid_z + 3): (mid_z + 8)], axis=2)  # ct_5_mid
        model_input[4, :, :] = np.average(rescaled_ct[:, :, (mid_z + 8): (mid_z + 13)], axis=2)  # ct_5_nex
        training_sample["model_input"] = model_input

        # ground truth for the model outputs, i.e., five ct_1
        model_gt = np.zeros([5, 512, 512])
        model_gt[0, :, :] = rescaled_ct[:, :, mid_z - 2]
        model_gt[1, :, :] = rescaled_ct[:, :, mid_z - 1]
        model_gt[2, :, :] = rescaled_ct[:, :, mid_z]
        model_gt[3, :, :] = rescaled_ct[:, :, mid_z + 1]
        model_gt[4, :, :] = rescaled_ct[:, :, mid_z + 2]
        training_sample["model_gt"] = model_gt

        # penalty weights:
        penalty_weight = np.zeros([5, 512, 512])
        penalty_weight[0, :, :] = final_penalty_mask[:, :, mid_z - 2]
        penalty_weight[1, :, :] = final_penalty_mask[:, :, mid_z - 1]
        penalty_weight[2, :, :] = final_penalty_mask[:, :, mid_z]
        penalty_weight[3, :, :] = final_penalty_mask[:, :, mid_z + 1]
        penalty_weight[4, :, :] = final_penalty_mask[:, :, mid_z + 2]
        training_sample["loss_weight"] = penalty_weight

        return training_sample

    training_sample_list = []

    for z in range(z_min, z_max, interval):
        training_sample_list.append((z, generate_one_sample(z)))

    return training_sample_list


def pipeline_process(dict_rescaled_ct, dict_lung_mask, dict_artery_1, dict_artery_2, dict_vein_1, dict_vein_2,
                     dict_airway_mask, dict_heart_mask, dict_list_lesion, dict_save):
    array_name_list = os.listdir(dict_rescaled_ct)
    print("There are:", len(array_name_list), "rescaled arrays")

    sample_name_list = os.listdir(dict_save)
    processed_array_name_set = set()
    for sample_name in sample_name_list:
        processed_array_name_set.add(sample_name[:-9])
    print(processed_array_name_set)

    processed = 0
    for array_name in array_name_list:
        print("processing:", array_name, processed, '/', len(array_name_list))
        if array_name[:-4] in processed_array_name_set:
            print('processed')
            processed += 1
            continue

        rescaled_ct = np.load(os.path.join(dict_rescaled_ct, array_name))
        lung_mask = np.load(os.path.join(dict_lung_mask, array_name[:-1] + 'z'))['arr_0']
        artery_1 = np.load(os.path.join(dict_artery_1, array_name[:-1] + 'z'))['arr_0']
        artery_2 = np.load(os.path.join(dict_artery_2, array_name[:-1] + 'z'))['arr_0']
        # blood_mask = np.load(os.path.join(dict_blood_mask, array_name[:-1] + 'z'))['array']
        # artery_mask = select_region(artery_mask, num=1)
        vein_1 = np.load(os.path.join(dict_vein_1, array_name[:-1] + 'z'))['arr_0']
        vein_2 = np.load(os.path.join(dict_vein_2, array_name[:-1] + 'z'))['arr_0']
        # vein_mask = select_region(vein_mask, num=1)
        blood_mask = np.array(artery_1 + artery_2 + vein_1 + vein_2 > 0.5, "float32")
        airway_mask = np.load(os.path.join(dict_airway_mask, array_name[:-1] + 'z'))['arr_0']
        heart_mask = np.load(os.path.join(dict_heart_mask, array_name[:-1] + 'z'))['arr_0']
        lesion_mask = np.zeros([512, 512, 512], 'float32')
        for lesion_dict in dict_list_lesion:
            lesion_mask = lesion_mask + np.load(os.path.join(lesion_dict, array_name[:-1] + 'z'))['array']
        if len(dict_list_lesion) > 0:
            lesion_mask = np.clip(lesion_mask, 0, 1)

        # training_sample_list in [(mid_z, sample), ...]
        training_sample_list = slice_one_rescaled_ct(rescaled_ct, lung_mask, blood_mask, airway_mask,
                                                     lesion_mask, heart_mask)
        print("there are", len(training_sample_list), 'samples')

        for item in training_sample_list:
            sample_name = array_name[:-4] + '_'
            mid_z = item[0]
            assert 0 <= mid_z < 10000
            if 1000 <= mid_z < 10000:
                sample_name = sample_name + str(mid_z)
            elif 100 <= mid_z < 1000:
                sample_name = sample_name + '0' + str(mid_z)
            elif 10 <= mid_z < 100:
                sample_name = sample_name + '00' + str(mid_z)
            else:
                sample_name = sample_name + '000' + str(mid_z)

            Functions.save_np_array(dict_save, sample_name, item[1], compress=True)

        processed += 1


if __name__ == '__main__':
    pipeline_process('/Artery_Vein/rescaled_ct',
                     '/Artery_Vein/mask/lung',
                     "/Artery_Vein/mask/artery_1",
                     "/Artery_Vein/mask/artery_2",
                     "/Artery_Vein/mask/vein_1",
                     "/Artery_Vein/mask/vein_2",
                     '/Artery_Vein/mask/airway',
                     '/Artery_Vein/mask/heart',
                     [],
                     dict_save='/home/chuy/Artery_Vein_Upsampling/upsampling_2')
