function [acc, roc, MI] = runLinearDiscriminant(x_trn,y_trn,x_test,y_test,params)

Mdl = fitcdiscr(x_trn,y_trn,'DiscrimType',params.DiscrimType);
[label, score] = predict(Mdl, x_test);
acc = sum(label == y_test) / numel(label);
rocObj = rocmetrics(y_test, score, Mdl.ClassNames);
roc = mean(rocObj.AUC);
MI = helper.getMI_binary(label,y_test);

end