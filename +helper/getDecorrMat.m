function [x_decorr] = getDecorrMat(x,y)
    x_decorr = shuffleData(x, y);
end

function x_decorr = shuffleData(x, y)
    x_decorr = nan(size(x));
    uY = unique(y);
    
    for i = 1:length(uY)
        idx = find(y == uY(i));
        for celli = 1:size(x, 2)
            idx_rand = idx(randperm(length(idx)));
            x_decorr(idx, celli) = x(idx_rand, celli);
        end
    end
end
