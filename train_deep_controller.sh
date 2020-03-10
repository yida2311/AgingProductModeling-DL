export CUDA_VISIBLE_DEVICES=3
python train_deep_controller.py \
--n_class 12 \
--lr 0.1 \
--weight_decay 1e-4 \
--momentum 0.0 \
--epoch 150 \
--data_path "/remote-home/ldy/data/controller/controller-3.8" \
--meta_path "/remote-home/ldy/data/controller/trainval.csv" \
--model_path "/remote-home/ldy/AgingProductModeling-DL/results/saved_models/" \
--log_path "/remote-home/ldy/AgingProductModeling-DL/results/logs/" \
--task_name "resnet_controller" \
--batch_size 4 \
