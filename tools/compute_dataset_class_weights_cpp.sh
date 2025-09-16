g++ -std=c++17 compute_dataset_class_weights.cpp -o compute_weights -lpng -lz `pkg-config --cflags --libs opencv4`

./compute_weights \
    --use_custom_path \
    --num_classes 124 \
    --dataset_dir '/home/robert.breslin/datasets/mapillary_vistas/training/v2.0/labels' \
    --suffix '.png'