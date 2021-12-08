python source/evaluate.py \
--weights ./result/runs_human_attributes_1/last.pt \
--logfile human_attribute_project/evaluate_result_1.txt \
--data config/human_attribute_1/data_config.yaml \
--cfg config/human_attribute_1/train_config.yaml \
--batch_size 128 \
--device cuda:0