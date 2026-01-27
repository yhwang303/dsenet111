if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Forecasting" ]; then
    mkdir ./logs/Forecasting
fi

root_path_name=./ohio
data_path_name=train_2018
model_id_name=ohio1
data_name=ohio

gpu=0
random_seed=2021
model_name=DSENet
label_len=48
seq_len=$(($label_len*2))

for pred_len in 6
do
  python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_${label_len}_${pred_len} \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 2 \
    --m_patch_len 2 \
    --m_stride 2 \
    --e_layers 3 \
    --n_heads 8 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 6 \
    --stride 4 \
    --local_ws 5 \
    --concat 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --patience 3 \
    --pct_start 0.2 \
    --des 'Cross_Subject_Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/${model_name}_${model_id_name}_${seq_len}_${label_len}_${pred_len}.log
done
for pred_len in 12
do
  python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_${label_len}_${pred_len} \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 2 \
    --m_patch_len 2 \
    --m_stride 2 \
    --e_layers 3 \
    --n_heads 8 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 8 \
    --stride 4 \
    --local_ws 5 \
    --concat 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --patience 3 \
    --pct_start 0.2 \
    --des 'Cross_Subject_Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/${model_name}_${model_id_name}_${seq_len}_${label_len}_${pred_len}.log
done

for pred_len in 24
do
  python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$label_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 6 \
    --m_patch_len 12 \
    --m_stride 8 \
    --e_layers 3 \
    --n_heads 8 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 2 \
    --stride 1 \
    --local_ws 3 \
    --concat 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --patience 3 \
    --pct_start 0.2\
    --des 'Cross_Subject_Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$label_len'_'$pred_len.log 
done


for pred_len in 48
do
  python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$label_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 6 \
    --m_patch_len 12 \
    --m_stride 8 \
    --e_layers 3 \
    --n_heads 8 \
    --d_model 64 \
    --d_ff 128 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 2 \
    --stride 1 \
    --local_ws 3 \
    --concat 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --patience 3 \
    --pct_start 0.2\
    --des 'Cross_Subject_Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$label_len'_'$pred_len.log 
done


for pred_len in 96
do
  python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$label_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 6 \
    --m_patch_len 36 \
    --m_stride 24 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 2 \
    --stride 1 \
    --local_ws 3 \
    --concat 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --patience 3 \
    --pct_start 0.2\
    --des 'Cross_Subject_Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$label_len'_'$pred_len.log 
done

for pred_len in 192
do
  python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$label_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 6 \
    --m_patch_len 36 \
    --m_stride 24 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 2 \
    --stride 1 \
    --local_ws 3 \
    --concat 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --patience 3 \
    --pct_start 0.2\
    --des 'Cross_Subject_Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$label_len'_'$pred_len.log 
done

for pred_len in 288
do
  python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$label_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features S \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 6 \
    --m_patch_len 36 \
    --m_stride 24 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 2 \
    --stride 1 \
    --local_ws 3 \
    --concat 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --patience 3 \
    --pct_start 0.2\
    --des 'Cross_Subject_Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$label_len'_'$pred_len.log 
done