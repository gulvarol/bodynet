function fig_joints3D(joints3D, filename, vws)

assert(size(joints3D, 1) == 3);

joints3D = bsxfun(@minus, joints3D, min(joints3D, [], 2));

minlim = min(joints3D, [], 2);
maxlim = max(joints3D, [], 2);
minl = min(minlim);
maxl = max(maxlim);

joints3D(1, :) = joints3D(1, :) - (minl - minlim(1));
joints3D(2, :) = joints3D(2, :) + (maxl - maxlim(2));

draw3DPose(joints3D, 20, '-', true);
hold off;
axis equal;
xlim([minl maxl]); zlim([minl maxl]); ylim([minl maxl]);
set(gca, 'Color', 'none')

set(gca, 'XTickLabels', []);
set(gca, 'YTickLabels', []);
set(gca, 'ZTickLabels', []);

%vws: [nviews x 3]
for viewno = 1:size(vws, 1)
    view( vws(viewno, :) );
    drawnow;
    if(~strcmp(filename, ''))
        export_fig(sprintf('%s_joints3D_v%d.png', filename, viewno), '-transparent');
        %saveas(gcf, sprintf('%s_joints3D_v%d.png', filename, viewno));
    end
end

end