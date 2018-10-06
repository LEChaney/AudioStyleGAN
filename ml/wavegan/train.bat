CALL activate tensorflow
python train_wavegan.py train .\train ^
--data_dir .\data\Final_Datasets\All ^
--data_first_window ^
--wavegan_genr_pp ^
--train_summary_secs 30
