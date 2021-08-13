""" Module __init__.py for dmt.metrics """

from dmt.metrics import medpy_metrics


### ------ #     Classification Metrics     # ----- ###
confusion_matrix = None
area_under_curve = None
f1_score = None

### ------ #     Localization Metrics     # ----- ###

### ------ #     Binary Array Surface Metrics     # ----- ###
dice_2d = dice_3d = medpy_metrics.dc
jaccard_2d = jaccard_3d = medpy_metrics.jc
hausdorff_2d = hausdorff_3d = medpy_metrics.hd
hausdorff95_2d = hausdorff_3d = medpy_metrics.hd95
precision_2d = precision_3d = medpy_metrics.precision
recall_2d = recall_3d = medpy_metrics.recall
true_positive_rate_2d = true_positive_rate_3d = medpy_metrics.true_positive_rate
true_negative_rate_2d = true_negative_rate_3d = medpy_metrics.true_negative_rate
sensitivity_2d = sensitivity_3d = medpy_metrics.sensitivity
specificity_2d = specificity_3d = medpy_metrics.specificity
positive_predicitive_value_2d = medpy_metrics.positive_predictive_value
positive_predicitive_value_3d = medpy_metrics.positive_predictive_value
average_surface_distance_2d = average_surface_distance_3d = medpy_metrics.asd
average_symmetric_surface_distance_2d = medpy_metrics.assd
average_symmetric_surface_distance_3d = medpy_metrics.assd
relative_absolute_distance_2d = medpy_metrics.ravd
relative_absolute_distance_3d = medpy_metrics.ravd

