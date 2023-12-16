function [acc, roc, MI, w, bias] = runSVM(x_trn,y_trn,x_test,y_test,bestC)

Mdl = fitcsvm(x_trn,y_trn,'BoxConstraint',bestC,'KernelFunction','linear');
[label, score] = predict(Mdl, x_test);
acc = sum(label == y_test) / numel(label);
rocObj = rocmetrics(y_test, score, Mdl.ClassNames);
roc = mean(rocObj.AUC);
MI = helper.getMI_binary(label,y_test);

w = Mdl.Beta;
bias = Mdl.Bias;

end