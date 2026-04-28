# Real Non-Square Stretch vs Letterbox Comparison

## Sample Selection

- Total selected real pairs: 12
- Seasons: fall, spring, summer, winter
- Tiers per season: 3

## Aggregate Summary

| variant | resize_mode | success_count | total | success_rate | avg_inliers | avg_inlier_ratio | avg_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aspect_stress | letterbox | 3 | 12 | 0.25 | 10.75 | 0.4069 | 1.1156 |
| aspect_stress | stretch | 2 | 12 | 0.1667 | 10.92 | 0.315 | 1.2883 |
| square | letterbox | 5 | 12 | 0.4167 | 26.58 | 0.2966 | 1.4538 |
| square | stretch | 5 | 12 | 0.4167 | 26.58 | 0.2966 | 1.4538 |

## Per-Case Results

| case_id | season | tier | variant | resize_mode | registration_reliable | selected_extractor | match_count | padding_filtered_count | inliers | inlier_ratio | rmse | bad_homography_shape | coordinate_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fall_tier_1_of_3_ROIs1970_fall_79_p361 | fall | tier_1_of_3 | square | stretch | True | superpoint | 19 | 0 | 9 | 0.4737 | 0.9459 | False | stretched_canvas |
| fall_tier_1_of_3_ROIs1970_fall_79_p361 | fall | tier_1_of_3 | square | letterbox | True | superpoint | 19 | 0 | 9 | 0.4737 | 0.9459 | False | letterbox_with_original_backprojection |
| fall_tier_1_of_3_ROIs1970_fall_79_p361_aspect | fall | tier_1_of_3 | aspect_stress | stretch | False | aliked | 23 | 0 | 7 | 0.3043 | 1.1397 | True | stretched_canvas |
| fall_tier_1_of_3_ROIs1970_fall_79_p361_aspect | fall | tier_1_of_3 | aspect_stress | letterbox | False | superpoint | 8 | 0 | 5 | 0.625 | 0.6809 | True | letterbox_with_original_backprojection |
| fall_tier_2_of_3_ROIs1970_fall_16_p821 | fall | tier_2_of_3 | square | stretch | True | superpoint | 106 | 0 | 39 | 0.3679 | 1.9594 | False | stretched_canvas |
| fall_tier_2_of_3_ROIs1970_fall_16_p821 | fall | tier_2_of_3 | square | letterbox | True | superpoint | 106 | 0 | 39 | 0.3679 | 1.9594 | False | letterbox_with_original_backprojection |
| fall_tier_2_of_3_ROIs1970_fall_16_p821_aspect | fall | tier_2_of_3 | aspect_stress | stretch | False | superpoint | 90 | 0 | 22 | 0.2444 | 1.8895 | False | stretched_canvas |
| fall_tier_2_of_3_ROIs1970_fall_16_p821_aspect | fall | tier_2_of_3 | aspect_stress | letterbox | True | aliked | 40 | 0 | 16 | 0.4 | 1.7642 | False | letterbox_with_original_backprojection |
| fall_tier_3_of_3_ROIs1970_fall_74_p485 | fall | tier_3_of_3 | square | stretch | False | superpoint | 42 | 0 | 6 | 0.1429 | 1.1984 | True | stretched_canvas |
| fall_tier_3_of_3_ROIs1970_fall_74_p485 | fall | tier_3_of_3 | square | letterbox | False | superpoint | 42 | 0 | 6 | 0.1429 | 1.1984 | True | letterbox_with_original_backprojection |
| fall_tier_3_of_3_ROIs1970_fall_74_p485_aspect | fall | tier_3_of_3 | aspect_stress | stretch | False | superpoint | 33 | 0 | 5 | 0.1515 | 0.4046 | True | stretched_canvas |
| fall_tier_3_of_3_ROIs1970_fall_74_p485_aspect | fall | tier_3_of_3 | aspect_stress | letterbox | False | superpoint | 14 | 2 | 5 | 0.3571 | 0.5549 | True | letterbox_with_original_backprojection |
| spring_tier_1_of_3_ROIs1158_spring_78_p215 | spring | tier_1_of_3 | square | stretch | False | aliked | 34 | 0 | 8 | 0.2353 | 1.2576 | False | stretched_canvas |
| spring_tier_1_of_3_ROIs1158_spring_78_p215 | spring | tier_1_of_3 | square | letterbox | False | aliked | 34 | 0 | 8 | 0.2353 | 1.2576 | False | letterbox_with_original_backprojection |
| spring_tier_1_of_3_ROIs1158_spring_78_p215_aspect | spring | tier_1_of_3 | aspect_stress | stretch | False | aliked | 16 | 0 | 11 | 0.6875 | 1.5547 | True | stretched_canvas |
| spring_tier_1_of_3_ROIs1158_spring_78_p215_aspect | spring | tier_1_of_3 | aspect_stress | letterbox | False | aliked | 34 | 0 | 6 | 0.1765 | 0.7541 | False | letterbox_with_original_backprojection |
| spring_tier_2_of_3_ROIs1158_spring_98_p240 | spring | tier_2_of_3 | square | stretch | True | superpoint | 111 | 0 | 38 | 0.3423 | 2.2296 | False | stretched_canvas |
| spring_tier_2_of_3_ROIs1158_spring_98_p240 | spring | tier_2_of_3 | square | letterbox | True | superpoint | 111 | 0 | 38 | 0.3423 | 2.2296 | False | letterbox_with_original_backprojection |
| spring_tier_2_of_3_ROIs1158_spring_98_p240_aspect | spring | tier_2_of_3 | aspect_stress | stretch | True | aliked | 71 | 0 | 26 | 0.3662 | 1.8922 | False | stretched_canvas |
| spring_tier_2_of_3_ROIs1158_spring_98_p240_aspect | spring | tier_2_of_3 | aspect_stress | letterbox | False | aliked | 38 | 0 | 14 | 0.3684 | 1.8936 | True | letterbox_with_original_backprojection |
| spring_tier_3_of_3_ROIs1158_spring_148_p522 | spring | tier_3_of_3 | square | stretch | False | aliked | 67 | 0 | 6 | 0.0896 | 0.6069 | True | stretched_canvas |
| spring_tier_3_of_3_ROIs1158_spring_148_p522 | spring | tier_3_of_3 | square | letterbox | False | aliked | 67 | 0 | 6 | 0.0896 | 0.6069 | True | letterbox_with_original_backprojection |
| spring_tier_3_of_3_ROIs1158_spring_148_p522_aspect | spring | tier_3_of_3 | aspect_stress | stretch | False | superpoint | 49 | 0 | 6 | 0.1224 | 0.5614 | False | stretched_canvas |
| spring_tier_3_of_3_ROIs1158_spring_148_p522_aspect | spring | tier_3_of_3 | aspect_stress | letterbox | False | aliked | 4 | 0 | 4 | 1 | 0 | True | letterbox_with_original_backprojection |
| summer_tier_1_of_3_summer_img_temperate_p1914 | summer | tier_1_of_3 | square | stretch | False | superpoint | 0 | 0 | 0 | 0 | None | True | stretched_canvas |
| summer_tier_1_of_3_summer_img_temperate_p1914 | summer | tier_1_of_3 | square | letterbox | False | superpoint | 0 | 0 | 0 | 0 | None | True | letterbox_canvas_no_homography |
| summer_tier_1_of_3_summer_img_temperate_p1914_aspect | summer | tier_1_of_3 | aspect_stress | stretch | False | superpoint | 19 | 0 | 8 | 0.4211 | 1.5323 | True | stretched_canvas |
| summer_tier_1_of_3_summer_img_temperate_p1914_aspect | summer | tier_1_of_3 | aspect_stress | letterbox | False | aliked | 3 | 0 | 0 | 0 | None | True | letterbox_canvas_no_homography |
| summer_tier_2_of_3_ROIs1868_summer_57_p884 | summer | tier_2_of_3 | square | stretch | False | superpoint | 45 | 0 | 11 | 0.2444 | 1.5848 | True | stretched_canvas |
| summer_tier_2_of_3_ROIs1868_summer_57_p884 | summer | tier_2_of_3 | square | letterbox | False | superpoint | 45 | 0 | 11 | 0.2444 | 1.5848 | True | letterbox_with_original_backprojection |
| summer_tier_2_of_3_ROIs1868_summer_57_p884_aspect | summer | tier_2_of_3 | aspect_stress | stretch | True | aliked | 30 | 0 | 10 | 0.3333 | 1.3216 | False | stretched_canvas |
| summer_tier_2_of_3_ROIs1868_summer_57_p884_aspect | summer | tier_2_of_3 | aspect_stress | letterbox | False | superpoint | 21 | 0 | 7 | 0.3333 | 1.3939 | False | letterbox_with_original_backprojection |
| summer_tier_3_of_3_ROIs1868_summer_41_p174 | summer | tier_3_of_3 | square | stretch | True | superpoint | 292 | 0 | 167 | 0.5719 | 1.8754 | False | stretched_canvas |
| summer_tier_3_of_3_ROIs1868_summer_41_p174 | summer | tier_3_of_3 | square | letterbox | True | superpoint | 292 | 0 | 167 | 0.5719 | 1.8754 | False | letterbox_with_original_backprojection |
| summer_tier_3_of_3_ROIs1868_summer_41_p174_aspect | summer | tier_3_of_3 | aspect_stress | stretch | False | aliked | 87 | 0 | 15 | 0.1724 | 1.8384 | False | stretched_canvas |
| summer_tier_3_of_3_ROIs1868_summer_41_p174_aspect | summer | tier_3_of_3 | aspect_stress | letterbox | True | superpoint | 42 | 0 | 13 | 0.3095 | 1.7201 | False | letterbox_with_original_backprojection |
| winter_tier_1_of_3_ROIs2017_winter_9_p18 | winter | tier_1_of_3 | square | stretch | False | superpoint | 10 | 0 | 5 | 0.5 | 1.0773 | True | stretched_canvas |
| winter_tier_1_of_3_ROIs2017_winter_9_p18 | winter | tier_1_of_3 | square | letterbox | False | superpoint | 10 | 0 | 5 | 0.5 | 1.0773 | True | letterbox_with_original_backprojection |
| winter_tier_1_of_3_ROIs2017_winter_9_p18_aspect | winter | tier_1_of_3 | aspect_stress | stretch | False | aliked | 29 | 0 | 9 | 0.3103 | 1.3892 | True | stretched_canvas |
| winter_tier_1_of_3_ROIs2017_winter_9_p18_aspect | winter | tier_1_of_3 | aspect_stress | letterbox | False | superpoint | 11 | 0 | 6 | 0.5455 | 0.8215 | True | letterbox_with_original_backprojection |
| winter_tier_2_of_3_ROIs2017_winter_37_p189 | winter | tier_2_of_3 | square | stretch | True | aliked | 44 | 0 | 22 | 0.5 | 1.9809 | False | stretched_canvas |
| winter_tier_2_of_3_ROIs2017_winter_37_p189 | winter | tier_2_of_3 | square | letterbox | True | aliked | 44 | 0 | 22 | 0.5 | 1.9809 | False | letterbox_with_original_backprojection |
| winter_tier_2_of_3_ROIs2017_winter_37_p189_aspect | winter | tier_2_of_3 | aspect_stress | stretch | False | aliked | 18 | 0 | 7 | 0.3889 | 1.0296 | True | stretched_canvas |
| winter_tier_2_of_3_ROIs2017_winter_37_p189_aspect | winter | tier_2_of_3 | aspect_stress | letterbox | True | superpoint | 89 | 0 | 47 | 0.5281 | 1.6531 | False | letterbox_with_original_backprojection |
| winter_tier_3_of_3_ROIs2017_winter_34_p246 | winter | tier_3_of_3 | square | stretch | False | superpoint | 88 | 0 | 8 | 0.0909 | 1.2758 | False | stretched_canvas |
| winter_tier_3_of_3_ROIs2017_winter_34_p246 | winter | tier_3_of_3 | square | letterbox | False | superpoint | 88 | 0 | 8 | 0.0909 | 1.2758 | False | letterbox_with_original_backprojection |
| winter_tier_3_of_3_ROIs2017_winter_34_p246_aspect | winter | tier_3_of_3 | aspect_stress | stretch | False | aliked | 18 | 0 | 5 | 0.2778 | 0.9067 | True | stretched_canvas |
| winter_tier_3_of_3_ROIs2017_winter_34_p246_aspect | winter | tier_3_of_3 | aspect_stress | letterbox | False | aliked | 25 | 0 | 6 | 0.24 | 1.0352 | False | letterbox_with_original_backprojection |
