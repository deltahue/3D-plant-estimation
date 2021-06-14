#!/bin/bash
echo "Masking pointcloud: "

DATASET_PATH=$1

if [ ! -f "$DATASET_PATH/dense/orig_fused.ply" ]; then
    echo "Renaming"
    cp $DATASET_PATH/dense/fused.ply $DATASET_PATH/dense/orig_fused.ply
    cp $DATASET_PATH/dense/meshed-poisson.ply $DATASET_PATH/dense/orig_meshed-poisson.ply
    cp $DATASET_PATH/dense/meshed-delaunay.ply $DATASET_PATH/dense/orig_meshed-delaunay.ply
fi

colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply

colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply
