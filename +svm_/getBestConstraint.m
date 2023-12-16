function [acc_best, boxConstraints] = getBestConstraint(X,Y,params)

uY = unique(Y);
cv = cvpartition(Y,"KFold",params.nFold_hyperparam,"Stratify",true);

acc_ = [];
for boxi = 1:length(params.BoxConstraint_range)
    acc =[];
    for cvi = 1:cv.NumTestSets
        x_trn = X(cv.training(cvi), :);
        x_test = X(cv.test(cvi), :);
        y_trn = Y(cv.training(cvi));
        y_test = Y(cv.test(cvi));

        t = templateSVM('BoxConstraint', 10^params.BoxConstraint_range(boxi), 'KernelFunction', 'linear', 'Standardize', false);
        
        if length(uY) > 2
            Mdl = fitcecoc(x_trn, y_trn, 'Learners', t);
        else
            Mdl = fitcsvm(x_trn, y_trn, 'BoxConstraint', 10^params.BoxConstraint_range(boxi), 'KernelFunction', 'linear');
        end

        [label, score] = predict(Mdl, x_test);
        acc(cvi) = sum(label == y_test) / numel(label);
        rocObj = rocmetrics(y_test, score, Mdl.ClassNames);
        roc(cvi) = mean(rocObj.AUC);
    end
    acc_ = mean(acc);
end

[acc_best, idx] = max(acc_);
boxConstraints = 10^params.BoxConstraint_range(idx);





end