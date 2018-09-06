function preprocess_surreal_fillvoxels(infopath)
    voxelspath = [infopath(1:end-8) 'voxels.mat'];
    V = load(voxelspath);
    N = numel(fieldnames(V));
    for i = 1:N
        eval(sprintf('vsurface = V.voxels_%d;', i));
        eval(sprintf('voxelsfill_%d = imfill(vsurface,''holes'');', i));
    end
    save([voxelspath(1:end-4) 'fill.mat'], 'voxelsfill_*');
end
