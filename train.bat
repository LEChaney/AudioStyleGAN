CALL activate tensorflow
python train_wavegan.py train .\train ^
--data_dir .\data\Final_Datasets\All ^
--data_first_window ^
--train_batch_size 128 ^
--wavegan_dim 32 ^
--train_summary_secs 5
