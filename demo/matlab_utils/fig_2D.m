function fig_2D(joints2D, segm, filename)

cm = segmColorMap();
cm = imresize(cm,[256, 3], 'nearest');
x = double(squeeze(segm));
img = ind2rgb(uint8(255*imshow_norm(x, [1 15])), cm);
imshow(uint8(255*img))
hold on;
draw2DPose(joints2D, 3, '-', true);
hold off;

if(~strcmp(filename, ''))
    export_fig(sprintf('%s_2D.png', filename));
    %saveas(gcf, sprintf('%s_2D.png', filename));
end

end