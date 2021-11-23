python source/evaluate.py \
--weights ./result/runs_human_attributes_4/best.pt \
--logfile human_attribute_evaluate.txt \
--data config/human_attribute_4/data_config.yaml \
--batch_size 64 \
--device cuda:0