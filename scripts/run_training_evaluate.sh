python preparing/prepare_dataset.py

python training/fasttext_training.py
python training/movies_seq_mt_kfold_training.py
python training/lgb_2_training.py
python training/lightfm_training.py

CUDA_VISIBLE_DEVICES=0 python predicting/predict_lightgbm.py
CUDA_VISIBLE_DEVICES=0 python predicting/predict_blend.py
