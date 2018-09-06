function fig_smpl(points, filename, vws, faces, facecolor)
if nargin < 5
    facecolor = [160 160 160]/255;
end
cla;
pointsdraw = points(:, [1 3 2]);
pointsdraw(:, 3) = -pointsdraw(:, 3);
light
light

patch('Faces', faces, 'Vertices', pointsdraw, 'FaceColor',facecolor, 'LineStyle', 'none', 'FaceLighting', 'gouraud', 'DiffuseStrength', 0.5)
axis equal;

set(gca, 'XTickLabels', []);
set(gca, 'YTickLabels', []);
set(gca, 'ZTickLabels', []);

minlim = min(pointsdraw, [], 1);
maxlim = max(pointsdraw, [], 1);

xlim([-0.5 0.8]); zlim([-0.8 1.2]); ylim([-0.5 0.3]);
minl = min(minlim);
maxl = max(maxlim);
xlim([minl maxl]); zlim([minl maxl]); ylim([minl maxl]);

set(gca, 'Color', 'none')
axis off;

%vws: [nviews x 3]
for viewno = 1:size(vws, 1)
    view( vws(viewno, :) );
    drawnow;
    if(~strcmp(filename, ''))
        export_fig(sprintf('%s_smpl_v%d.png', filename, viewno), '-transparent');
        %saveas(gcf, sprintf('%s_smpl_v%d.png', filename, viewno));
    end
end

end