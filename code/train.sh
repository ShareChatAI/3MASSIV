#mode="vs_as"
#mode="vs_2as" # Video + Both Audio
mode="vs" # Only video
#mode="as" # Only audio
#mode="2as" # Both audio
featloc="./features/3massiv_resnext50/"
audio_featloc="./features/3massiv_clsril/"
audio_featloc_second="./features/3massiv_vggish/feats/"
phase="train"
train_file_path="../data/3massiv_concept_train.csv"
val_file_path="../data/3massiv_concept_val.csv"

CUDA_VISIBLE_DEVICES=0 python3 run_main.py --gpus -1 \
	--train_file $train_file_path \
	--val_file $val_file_path \
	--distributed_backend ddp \
	--check_val_every_n_epoch 1 \
	--run 09\
	--lr 5e-3 \
	--max_epochs 500 \
	--video_location $featloc \
	--audio_location $audio_featloc \
	--audio_featloc_second $audio_featloc_second \
	--mode $mode \
	--phase $phase \



