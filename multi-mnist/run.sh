CUDA_VISIBLE_DEVICES=0 python train.py --dset "multi_fashion_and_mnist" \
    --seed 2 \
    --adaptive True \
    --rho 2 \
    --c 0.4 \
    --method cagrad \
    --rho_eval \
    --n_epochs 200 &

CUDA_VISIBLE_DEVICES=1 python train.py --dset "multi_mnist" \
    --seed 2 \
    --adaptive True \
    --rho 2 \
    --c 0.4 \
    --method cagrad \
    --rho_eval   \
    --n_epochs 200 &

CUDA_VISIBLE_DEVICES=2 python train.py --dset "multi_fashion" \
    --seed 2 \
    --adaptive True \
    --rho 2 \
    --c 0.4 \
    --method cagrad \
    --rho_eval  \
    --n_epochs 200 &

wait

CUDA_VISIBLE_DEVICES=0 python train.py --dset "multi_fashion_and_mnist" \
    --seed 2 \
    --adaptive True \
    --rho 0 \
    --c 0.4 \
    --method cagrad \
    --rho_eval   \
    --n_epochs 200 &

CUDA_VISIBLE_DEVICES=1 python train.py --dset "multi_mnist" \
    --seed 2 \
    --adaptive True \
    --rho 0 \
    --c 0.4 \
    --method cagrad \
    --rho_eval   \
    --n_epochs 200 &

CUDA_VISIBLE_DEVICES=2 python train.py --dset "multi_fashion" \
    --seed 2 \
    --adaptive True \
    --rho 0 \
    --c 0.4 \
    --method cagrad \
    --rho_eval   \
    --n_epochs 200 &

wait