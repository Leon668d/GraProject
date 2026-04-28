# Stretch vs Letterbox Comparison

| case_id | group | variant | resize_mode | registration_reliable | selected_extractor | match_count | padding_filtered_count | inliers | inlier_ratio | rmse | bad_homography_shape | coordinate_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| success_summer | success | square | stretch | True | superpoint | 633 | 0 | 434 | 0.6856 | 1.8318 | False | stretched_canvas |
| success_summer | success | square | letterbox | True | superpoint | 633 | 0 | 434 | 0.6856 | 1.8318 | False | letterbox_with_original_backprojection |
| success_winter | success | square | stretch | True | superpoint | 497 | 0 | 347 | 0.6982 | 1.7946 | False | stretched_canvas |
| success_winter | success | square | letterbox | True | superpoint | 497 | 0 | 347 | 0.6982 | 1.7946 | False | letterbox_with_original_backprojection |
| success_rescue_spring | success_rescue | square | stretch | True | aliked | 349 | 0 | 146 | 0.4183 | 1.9265 | False | stretched_canvas |
| success_rescue_spring | success_rescue | square | letterbox | True | aliked | 349 | 0 | 146 | 0.4183 | 1.9265 | False | letterbox_with_original_backprojection |
| boundary_winter | boundary | square | stretch | True | superpoint | 54 | 0 | 27 | 0.5 | 1.553 | False | stretched_canvas |
| boundary_winter | boundary | square | letterbox | True | superpoint | 54 | 0 | 27 | 0.5 | 1.553 | False | letterbox_with_original_backprojection |
| success_summer_aspect | success_aspect | aspect_stress | stretch | False | aliked | 32 | 0 | 6 | 0.1875 | 0.5171 | True | stretched_canvas |
| success_summer_aspect | success_aspect | aspect_stress | letterbox | False | superpoint | 28 | 0 | 10 | 0.3571 | 1.2905 | True | letterbox_with_original_backprojection |
| boundary_winter_aspect | boundary_aspect | aspect_stress | stretch | False | aliked | 23 | 0 | 6 | 0.2609 | 0.3686 | True | stretched_canvas |
| boundary_winter_aspect | boundary_aspect | aspect_stress | letterbox | True | superpoint | 17 | 1 | 9 | 0.5294 | 1.7764 | False | letterbox_with_original_backprojection |
