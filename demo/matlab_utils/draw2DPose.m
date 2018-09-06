function draw2DPose(J, lw, ls, pelvis)
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
% [x0 x1], [y0 y1]
line([J( 8, 1) J(9,  1)], [J( 8, 2) J( 9, 2)], 'Color', 'b', 'LineWidth', lw, 'LineStyle', ls); % 13 16 - 8 9 - head/neck
line([J( 8, 1) J(11, 1)], [J( 8, 2) J(11, 2)], 'Color', 'g', 'LineWidth', lw*1.2, 'LineStyle', ls); % 13 18 - 8 11 - neck/LShoul
line([J( 8, 1) J(10, 1)], [J( 8, 2) J(10, 2)], 'Color', 'r', 'LineWidth', lw, 'LineStyle', ls); % 13 17 - 8 10 - neck/RShoul
line([J(10, 1) J(12, 1)], [J(10, 2) J(12, 2)], 'Color', 'r', 'LineWidth', lw, 'LineStyle', ls); % 17 19 - 10 12 - RShoul/Relbow
line([J(12, 1) J(14, 1)], [J(12, 2) J(14, 2)], 'Color', 'r', 'LineWidth', lw, 'LineStyle', ls); % 19 21 - 12 14 - Relbow/Rhand
line([J(11, 1) J(13, 1)], [J(11, 2) J(13, 2)], 'Color', 'g', 'LineWidth', lw*1.2, 'LineStyle', ls); % 18 20 - 11 13 - LShoul/Lelbow
line([J(13, 1) J(15, 1)], [J(13, 2) J(15, 2)], 'Color', 'g', 'LineWidth', lw*1.2, 'LineStyle', ls); % 20 22 - 13 15 - Lelbow/Lhand
if(pelvis)
    line([J( 8, 1) J( 1, 1)], [J( 8, 2) J( 1, 2)], 'Color', 'b', 'LineWidth', lw, 'LineStyle', ls); % 13 1 - 8 1 - neck/pelvis
    line([J( 1, 1) J( 2, 1)], [J( 1, 2) J( 2, 2)], 'Color', 'r', 'LineWidth', lw*1.2, 'LineStyle', ls); % 1 2 - 1 2 - pelvis/Lleg
    line([J( 1, 1) J( 3, 1)], [J( 1, 2) J( 3, 2)], 'Color', 'g', 'LineWidth', lw, 'LineStyle', ls); % 1 3 - 1 3 - pelvis/Rleg
end
line([J( 2, 1) J( 4, 1)], [J( 2, 2) J( 4, 2)], 'Color', 'r', 'LineWidth', lw*1.2, 'LineStyle', ls); % 2 5 - 2 4 - Lleg/Lknee
line([J( 4, 1) J( 6, 1)], [J( 4, 2) J( 6, 2)], 'Color', 'r', 'LineWidth', lw*1.2, 'LineStyle', ls); % 5 8 - 4 6 - Lknee/Lfoot
line([J( 3, 1) J( 5, 1)], [J( 3, 2) J( 5, 2)], 'Color', 'g', 'LineWidth', lw, 'LineStyle', ls); % 3 6 - 3 5 - Rleg/Lknee
line([J( 5, 1) J( 7, 1)], [J( 5, 2) J( 7, 2)], 'Color', 'g', 'LineWidth', lw, 'LineStyle', ls); % 6 9 - 5 7 - Rknee/Rfoot

%1     2     3     5     6     8     9    13    16    17    18    19    20    21    22
%1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
end
