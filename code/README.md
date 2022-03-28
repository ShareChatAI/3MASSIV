# Training the models:

```
bash train.sh
```
Please set the ```mode``` in the ```train.sh``` for training the model with desired modalities.

# Testing a pretrained model:

```
bash test.sh
```
Please set the ```mode``` and model path ```model_path``` in the ```test.sh``` for testing the model.

Note: Both train and test scripts expect the path to the visual and audio features which can be extracted using the following steps.

# Feature Extraction:

**Visual - ResNext**: We use the repository by Kenshohara (https://github.com/kenshohara/3D-ResNets-PyTorch) for extracting the visual features from our videos. We extract frames from our video at 25fps and scale the shorter side to 240, while maintaining the aspect ratio. <br />

Please refer the following code for details of frame extraction from videos:<br />
(https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/util_scripts/generate_video_jpgs.py)

**Audio - CLSRIL**: We use ffmpeg to extract audio wav files from our videos and use the OpenSpeech repository (https://github.com/Open-Speech-EkStep/vakyansh-wav2vec2-experimentation) for extracting the audio features.

**Audio - VGG**: We use (https://github.com/tensorflow/models/tree/master/research/audioset/vggish) for extracting VGG features from wav files.
