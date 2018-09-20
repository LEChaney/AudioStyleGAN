CALL activate tensorflow
python train_wavegan.py train .\train ^
--data_dir .\data\Final_Datasets\All ^
--data_first_window ^
--wavegan_genr_upsample nn ^
--wavegan_disc_phaseshuffle 0 ^
--train_summary_secs 5
