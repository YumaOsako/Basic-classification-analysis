function [ax] = figModule
ax = gca;
ax.FontName = 'Arial';
ax.FontSize = 12;
ax.LabelFontSizeMultiplier = 1.5;
ax.LineWidth = 1;
ax.TickDir = 'out';
box off;
end