import torch

# data = torch.load("data/BDD-Detection/retinanet/retinanet_R_50_FPN_1x/random_seed_0/inference/bdd_val/anchor_statistics/preprocessed_predicted_instances_0.0.pth")

# data = torch.load("BDD_DATASET_ROOT/labels/preprocessed_gt_instances.pth")

# print(f"File: BDD_DATASET_ROOT/labels/preprocessed_gt_instances.pth, keys: {data.keys()}")

# data = torch.load("data/BDD-Detection/retinanet/retinanet_R_50_FPN_1x/random_seed_0/inference/bdd_val/anchor_statistics/preprocessed_predicted_instances_0.0.pth")

# print(f"File: data/BDD-Detection/retinanet/retinanet_R_50_FPN_1x/random_seed_0/inference/bdd_val/anchor_statistics/preprocessed_predicted_instances_0.0.pth, keys: {data.keys()}")

# data = torch.load("data/BDD-Detection/retinanet/retinanet_R_50_FPN_1x/random_seed_0/inference/bdd_val/anchor_statistics/matched_results_0.1_0.7_0.0.pth")

# print(f"File: data/BDD-Detection/retinanet/retinanet_R_50_FPN_1x/random_seed_0/inference/bdd_val/anchor_statistics/matched_results_0.1_0.7_0.0.pth, keys: {data.keys()}")

# data = torch.load("data/BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout/random_seed_1/inference/bdd_val/bayes_od_mc_dropout/matched_results_0.1_0.7_0.0.pth")
data = torch.load("data/BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout/random_seed_1/inference/bdd_val/bayes_od_mc_dropout/preprocessed_predicted_instances_0.0.pth")

print(data.keys())