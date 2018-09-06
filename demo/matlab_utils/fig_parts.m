function fig_parts(partclass, filename, vws)

res = 128;
partcolors = [1 0 1; 1 1 0; 0 1 1; 0 1 0; 1 0 0; 0 0 1];
cla;
x = 1:res;
y = 1:res;
z = 1:res;
[X,Y,Z] = meshgrid(x,y,z);
fv = cell(7);
for pp = 2:7
    fv{pp} = isosurface(Y,X,Z,single(partclass == pp), 0.5);
    if(~isempty(fv{pp}.vertices))
        fv{pp} = smoothpatch(fv{pp}, 1, 5);

        fv{pp}.EdgeColor = 'none';
        fv{pp}.FaceColor = partcolors(pp-1, :);
        patch(fv{pp});
        fv{pp}.LineStyle = 'none';
        hold on;
    end
end
hold off;

axis equal;
set(gca, 'XTickLabels', []);
set(gca, 'YTickLabels', []);
set(gca, 'ZTickLabels', []);
axis off;

xlim([1 res]);
ylim([1 res]);
zlim([1 res]);

camlight
camlight
lighting gouraud

%vws: [nviews x 3]
for viewno = 1:size(vws, 1)
    view( vws(viewno, :) ); 
    drawnow;
    if(~strcmp(filename, ''))
        export_fig(sprintf('%s_parts_v%d.png', filename, viewno), '-transparent');
        %saveas(gcf, sprintf('%s_parts_v%d.png', filename, viewno));
    end
end
