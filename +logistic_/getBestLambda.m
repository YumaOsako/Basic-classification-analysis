function [acc_best, lambda] = getBestLambda(X,Y,params)

uY = unique(Y);
cv = cvpartition(Y,"KFold",params.nFold_hyperparam,"Stratify",true);

acc_ = [];
for li = 1:length(params.lambda_n)
    acc = [];
    for cvi = 1:cv.NumTestSets
        x_trn = X(cv.training(cvi), :);
        x_test = X(cv.test(cvi), :);
        y_trn = Y(cv.training(cvi));
        y_test = Y(cv.test(cvi));

        Mdl = fitclinear(x_trn,y_trn,"Lambda",10^(params.lambda_n(li)),'Learner','logistic',...
            'Regularization',params.regularization);

        [label, score] = predict(Mdl, x_test);
        acc(cvi) = sum(label == y_test) / numel(label);
        rocObj = rocmetrics(y_test, score, Mdl.ClassNames);
        roc(cvi) = mean(rocObj.AUC);
    end
    acc_(li) = mean(acc);
end

[acc_best, idx] = max(acc_);
lambda = 10^params.lambda_n(idx);





end