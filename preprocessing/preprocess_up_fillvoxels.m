function preprocess_up_fillvoxels(shapepath)
    voxels = load(shapepath, 'voxels');
    voxelsfill = imfill(voxels, 'holes');
    save(shapepath, 'voxelsfill', '-append');
end
