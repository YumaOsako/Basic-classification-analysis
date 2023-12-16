function [acc, roc, MI] = runMultiSVM(x_trn,y_trn,x_test,y_test,bestC)
%% Empirical data
t = templateSVM('BoxConstraint', bestC, 'KernelFunction', 'linear', 'Standardize', false);
Mdl = fitcecoc(x_trn, y_trn, 'Learners', t);

[label, score] = predict(Mdl, x_test);
acc = sum(label == y_test) / numel(label);
rocObj = rocmetrics(y_test, score, Mdl.ClassNames);
roc = mean(rocObj.AUC);
MI = helper.getMI(label,y_test);

end