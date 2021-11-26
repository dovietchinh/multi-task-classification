python source/evaluate.py \
--weights ./result/runs_elevator/last.pt \
--logfile human_attribute_project/evaluate_result.txt \
--data config/elevator_1/data_config.yaml \
--cfg config/elevator_1/train_config.yaml \
--batch_size 64 \
--device cuda:0