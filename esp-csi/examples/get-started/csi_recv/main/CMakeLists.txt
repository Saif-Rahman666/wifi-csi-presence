cmake_minimum_required(VERSION 4.0)
idf_component_register(
    SRCS "main.c" "csi_model.cpp" "ble_alert.c"
    INCLUDE_DIRS "."
    EMBED_FILES "cnn1d_int8.tflite"      # copy models/checkpoints/cnn1d_int8.tflite into main/
)
