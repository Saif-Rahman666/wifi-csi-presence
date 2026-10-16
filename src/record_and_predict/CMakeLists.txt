idf_component_register(
    SRCS "main.c" "csi_model.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES "cnn1d_int8.tflite"      # copy models/checkpoints/cnn1d_int8.tflite into main/
)
