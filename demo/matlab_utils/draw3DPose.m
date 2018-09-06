function draw3DPose(J, lw, ls, pelvis)

if nargin < 2
    lw = 5;
end
if nargin < 3
    ls = '-';
end
if nargin < 4
    pelvis = true;
end
%pelvis = false;
J = J';

lc = 'g'; % leftcolor
rc = 'r'; % rightcolor

% [x0 x1], [y0 y1], [z0 z1]
plot3([J( 8, 1) J(9,  1)], [J( 8, 2) J( 9, 2)], [J( 8, 3) J( 9, 3)], 'Color', 'b', 'LineWidth', lw, 'LineStyle', ls); % 13 16 - 8 9 - head/neck
hold on;
plot3([J( 8, 1) J(11, 1)], [J( 8, 2) J(11, 2)], [J( 8, 3) J(11, 3)], 'Color', lc, 'LineWidth', lw*1.2, 'LineStyle', ls); % 13 18 - 8 11 - neck/LShoul
plot3([J( 8, 1) J(10, 1)], [J( 8, 2) J(10, 2)], [J( 8, 3) J(10, 3)], 'Color', rc, 'LineWidth', lw, 'LineStyle', ls); % 13 17 - 8 10 - neck/RShoul
plot3([J(10, 1) J(12, 1)], [J(10, 2) J(12, 2)], [J(10, 3) J(12, 3)], 'Color', rc, 'LineWidth', lw, 'LineStyle', ls); % 17 19 - 10 12 - RShoul/Relbow
plot3([J(12, 1) J(14, 1)], [J(12, 2) J(14, 2)], [J(12, 3) J(14, 3)], 'Color', rc, 'LineWidth', lw, 'LineStyle', ls); % 19 21 - 12 14 - Relbow/Rhand
plot3([J(11, 1) J(13, 1)], [J(11, 2) J(13, 2)], [J(11, 3) J(13, 3)], 'Color', lc, 'LineWidth', lw*1.2, 'LineStyle', ls); % 18 20 - 11 13 - LShoul/Lelbow
plot3([J(13, 1) J(15, 1)], [J(13, 2) J(15, 2)], [J(13, 3) J(15, 3)], 'Color', lc, 'LineWidth', lw*1.2, 'LineStyle', ls); % 20 22 - 13 15 - Lelbow/Lhand
if(pelvis)
    plot3([J( 8, 1) J( 1, 1)], [J( 8, 2) J( 1, 2)], [J( 8, 3) J( 1, 3)], 'Color', 'b', 'LineWidth', lw, 'LineStyle', ls); % 13 1 - 8 1 - neck/pelvis
    plot3([J( 1, 1) J( 2, 1)], [J( 1, 2) J( 2, 2)], [J( 1, 3) J( 2, 3)], 'Color', rc, 'LineWidth', lw*1.2, 'LineStyle', ls); % 1 2 - 1 2 - pelvis/Lleg
    plot3([J( 1, 1) J( 3, 1)], [J( 1, 2) J( 3, 2)], [J( 1, 3) J( 3, 3)], 'Color', lc, 'LineWidth', lw, 'LineStyle', ls); % 1 3 - 1 3 - pelvis/Rleg
end
plot3([J( 2, 1) J( 4, 1)], [J( 2, 2) J( 4, 2)], [J( 2, 3) J( 4, 3)], 'Color', rc, 'LineWidth', lw*1.2, 'LineStyle', ls); % 2 5 - 2 4 - Lleg/Lknee
plot3([J( 4, 1) J( 6, 1)], [J( 4, 2) J( 6, 2)], [J( 4, 3) J( 6, 3)], 'Color', rc, 'LineWidth', lw*1.2, 'LineStyle', ls); % 5 8 - 4 6 - Lknee/Lfoot
plot3([J( 3, 1) J( 5, 1)], [J( 3, 2) J( 5, 2)], [J( 3, 3) J( 5, 3)], 'Color', lc, 'LineWidth', lw, 'LineStyle', ls); % 3 6 - 3 5 - Rleg/Lknee
plot3([J( 5, 1) J( 7, 1)], [J( 5, 2) J( 7, 2)], [J( 5, 3) J( 7, 3)], 'Color', lc, 'LineWidth', lw, 'LineStyle', ls); % 6 9 - 5 7 - Rknee/Rfoot

joint_col = [ 0 0 0.5];
scatter3(J(:, 1), J(:, 2), J(:, 3), 400, 'filled', 'o', 'MarkerEdgeColor', joint_col, 'MarkerFaceColor', joint_col, 'LineWidth', 13);

axis off;
hold off;
%1     2     3     5     6     8     9    13    16    17    18    19    20    21    22
%1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
end