#mode="vs_as"
mode="vs_2as" # Video + Both Audio
#mode="vs" # Only video
#mode="as" # Only audio
#mode="2as" # Both audio
featloc="/mnt/disks/3massiv/resnext/"
audio_featloc="/mnt/disks/3massiv/clsril/"
audio_featloc_second="/mnt/disks/3massiv/vggish/feats/"
phase="train"
train_file_path="../data/3massiv_train_dummy.csv"
val_file_path="../data/3massiv_val_dummy.csv"

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



