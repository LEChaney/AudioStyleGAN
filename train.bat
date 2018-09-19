CALL activate tensorflow
python train_wavegan.py train .\train ^
--data_dir .\data\Final_Datasets\All ^
--data_first_window ^
--train_batch_size 128 ^
--wavegan_disc_phaseshuffle 0 ^
--wavegan_genr_upsample nn ^
--train_summary_secs 60
