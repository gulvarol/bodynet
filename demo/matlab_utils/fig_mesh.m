function fig_mesh(pred_voxels, texture, rgb, filename, vws)
thrs = 0.5;
cla;
x = 1:128;
y = 1:128;
z = 1:128;
[X,Y,Z] = meshgrid(x,y,z);
fv = isosurface(Y,X,Z,pred_voxels,thrs);

if size(fv.vertices, 1) > 0
    fv.FaceColor = [160 160 160]/255;
    fv.LineStyle = 'none';
    fv.FaceLighting = 'gouraud';
    fv.DiffuseStrength= 0.5;
    
    if texture
        im = rgb;
        red = double(im(:,:,1));
        green = double(im(:,:,2));
        blue = double(im(:,:,3));
        [vcx,vcy] = meshgrid(1:128,1:128);
        vc = [vcx(:) vcy(:) red(:) green(:) blue(:)];
        voxcor = fv.vertices(:,[1 3]);
        voxcor(:, 2) = 128-voxcor(:, 2);
        cols = knnsearch(vc(:,1:2), voxcor);
        fv.FaceVertexCData = uint8(255*vc(cols,3:5));
        fv.FaceColor = 'interp';
    end
    
    patch(fv);
    
    axis equal;
    set(gca, 'XTickLabels', []);
    set(gca, 'YTickLabels', []);
    set(gca, 'ZTickLabels', []);
    
    axis off;
    view(18, 15);
    
    xlim([1 128]);
    ylim([1 128]);
    zlim([1 128]);
    
    camlight
    camlight
    lighting gouraud
end

%vws: [nviews x 3]
for viewno = 1:size(vws, 1)
    view( vws(viewno, :) ); 
    drawnow;
    if(~strcmp(filename, ''))
        export_fig(sprintf('%s_textured_v%d.png', filename, viewno), '-transparent');
        %saveas(gcf, sprintf('%s_textured_v%d.png', filename, viewno));
    end
end
