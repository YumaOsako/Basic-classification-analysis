%% data loading
% We will use iris dataset which contains meas (variable: #data * feartures)
% and species (label).

% comment out if multi-class svm is selected 

% load fisheriris
% species_ = grp2idx(species);

% comment out if multi-class svm is selected 
load fisheriris
inds = ~strcmp(species,'setosa');
meas = meas(inds,:);
species = species(inds);

species_ = grp2idx(species);


%% Params setting

params = struct;
params.method = 'svm'; % 'svm', 'logistic', 'naiveBayes', 'LD', 'svm multi'
param.decorr = true; % decoding using decorrelated data if true
params.nBoot = 100; % Number of repetition of cross validation
params.cv = 'KFold'; % 'KFold', 'Holdout', 'Leaveout'
params.nFold = 5; % Number of fold (k), required if  cv='kFold'
params.nPartition = 0.2; % Proportion of test set, required if  cv='Holdout'
params.BoxConstraint_range = -3:3; % range of gridserach of penalty, required if method = 'svm' or 'svm multi'
params.nFold_hyperparam = 3; % Number of fold (k) when gridseraching
params.lambda_n = 2:-2:-6; % Regularization term, required if method = 'logistic'
params.regularization = 'lasso'; % Regularization type, required if method = 'logistic'.  {'lasso','ridge'}
params.distribution = 'normal'; % Type of predictor distribution, required if method = 'naiveBayes'.  
                                % {'normal','kernel','mn','mvmn'}
params.DiscrimType = 'linear'; % Discrimination type, required if method = 'LD'.
                               % {'linear','diaglinear','pseudolinear','quadratic','diagquadratic','pseudoquadratic'}

%% data preprocess
% We MUST normalize predictors (X) before classification analysis

zMeas = zscore(meas);
[nData, nFeatures] = size(zMeas);

if param.decorr
    [zMeas_decorr] = helper.getDecorrMat(zMeas,species_);
end

%% initialization 

acc_boot = [];
roc_boot = [];
MI_boot = [];
w_boot = [];
bias_boot = [];

sacc_boot = [];
sroc_boot = [];
sMI_boot = [];
sw_boot = [];
sbias_boot = [];

%% Decoding 

for booti = 1:params.nBoot
    tic;
    % train-test split
    switch params.cv
        case 'KFold'
            cv = cvpartition(species,params.cv,params.nFold,"Stratify",true);
        case 'Holdout'
            cv = cvpartition(species,params.cv,params.nPartition,"Stratify",true);
        case 'Leaveout'
            cv = cvpartition(species,params.cv);
    end

    acc = [];
    roc = [];
    MI = [];
    w = [];
    bias = [];
    for cvi = 1:cv.NumTestSets
        x_trn = zMeas(cv.training(cvi),:);
        x_test = zMeas(cv.test(cvi),:);
        y_trn = species_(cv.training(cvi));
        y_test = species_(cv.test(cvi));
        y_trn_shuffled = y_trn(randperm(length(y_trn)));

        switch params.method
            case 'svm multi'
                [~,bestC] = svm_.getBestConstraint(x_trn,y_trn,params);
                [acc(cvi), roc(cvi), MI(cvi)] = svm_.runMultiSVM(x_trn,y_trn,x_test, y_test,bestC);
                [~,bestC] = svm_.getBestConstraint(x_trn,y_trn_shuffled,params);
                [sacc(cvi), sroc(cvi), sMI(cvi)] = svm_.runMultiSVM(x_trn,y_trn_shuffled,x_test, y_test,bestC);
            case 'svm'
                [~,bestC] = svm_.getBestConstraint(x_trn,y_trn,params);
                [acc(cvi), roc(cvi), MI(cvi), w(:,cvi), bias(cvi)] = svm_.runSVM(x_trn,y_trn,x_test, y_test,bestC);
                [~,bestC] = svm_.getBestConstraint(x_trn,y_trn_shuffled,params);
                [sacc(cvi), sroc(cvi), sMI(cvi), sw(:,cvi), sbias(cvi)] = svm_.runSVM(x_trn,y_trn_shuffled,x_test, y_test,bestC);
            case 'logistic'
                [~,bestLambda] = logistic_.getBestLambda(x_trn,y_trn,params);
                [acc(cvi), roc(cvi), MI(cvi), w(:,cvi), bias(cvi)] = logistic_.runLogisticClass(x_trn,y_trn,x_test, y_test,bestLambda,params);
                [~,bestLambda] = logistic_.getBestLambda(x_trn,y_trn_shuffled,params);
                [sacc(cvi), sroc(cvi), sMI(cvi), sw(:,cvi), sbias(cvi)] = logistic_.runLogisticClass(x_trn,y_trn_shuffled,x_test, y_test,bestLambda,params);
            case 'naiveBayes'
                [acc(cvi), roc(cvi), MI(cvi)] = naivebayes_.runNaiveBayes(x_trn,y_trn,x_test, y_test,params);
                [sacc(cvi), sroc(cvi), sMI(cvi)] = naivebayes_.runNaiveBayes(x_trn,y_trn_shuffled,x_test, y_test,params);
            case 'LD'
                [acc(cvi), roc(cvi), MI(cvi)] = ld_.runLinearDiscriminant(x_trn,y_trn,x_test, y_test,params);
                [sacc(cvi), sroc(cvi), sMI(cvi)] = ld_.runLinearDiscriminant(x_trn,y_trn_shuffled,x_test, y_test,params);
        end

        if param.decorr
            x_trn_decorr = zMeas_decorr(cv.training(cvi),:);
            x_test_decorr = zMeas_decorr(cv.test(cvi),:);

            switch params.method
                case 'svm multi'
                    [~,bestC] = svm_.getBestConstraint(x_trn_decorr,y_trn,params);
                    [acc_decorr(cvi), roc_decorr(cvi), MI_decorr(cvi)] = svm_.runMultiSVM(x_trn_decorr,y_trn,x_test_decorr, y_test,bestC);
                    [~,bestC] = svm_.getBestConstraint(x_trn_decorr,y_trn_shuffled,params);
                    [sacc_decorr(cvi), sroc_decorr(cvi), sMI_decorr(cvi)] = svm_.runMultiSVM(x_trn_decorr,y_trn_shuffled,x_test_decorr, y_test,bestC);
                case 'svm'
                    [~,bestC] = svm_.getBestConstraint(x_trn_decorr,y_trn,params);
                    [acc_decorr(cvi), roc_decorr(cvi), MI_decorr(cvi), w_decorr(:,cvi), bias_decorr(cvi)] = svm_.runSVM(x_trn_decorr,y_trn,x_test_decorr, y_test,bestC);
                    [~,bestC] = svm_.getBestConstraint(x_trn_decorr,y_trn_shuffled,params);
                    [sacc_decorr(cvi), sroc_decorr(cvi), sMI_decorr(cvi), sw_decorr(:,cvi), sbias_decorr(cvi)] = svm_.runSVM(x_trn_decorr,y_trn_shuffled,x_test_decorr, y_test,bestC);
                case 'logistic'
                    [~,bestLambda] = logistic_.getBestLambda(x_trn_decorr,y_trn,params);
                    [acc_decorr(cvi), roc_decorr(cvi), MI_decorr(cvi), w_decorr(:,cvi), bias_decorr(cvi)] = logistic_.runLogisticClass(x_trn_decorr,y_trn,x_test_decorr, y_test,bestLambda,params);
                    [~,bestLambda] = logistic_.getBestLambda(x_trn_decorr,y_trn_shuffled,params);
                    [sacc_decorr(cvi), sroc_decorr(cvi), sMI_decorr(cvi), sw_decorr(:,cvi), sbias_decorr(cvi)] = logistic_.runLogisticClass(x_trn_decorr,y_trn_shuffled,x_test_decorr, y_test,bestLambda,params);
                case 'naiveBayes'
                    [acc_decorr(cvi), roc_decorr(cvi), MI_decorr(cvi)] = naivebayes_.runNaiveBayes(x_trn_decorr,y_trn,x_test_decorr, y_test,params);
                    [sacc_decorr(cvi), sroc_decorr(cvi), sMI_decorr(cvi)] = naivebayes_.runNaiveBayes(x_trn_decorr,y_trn_shuffled,x_test_decorr, y_test,params);
                case 'LD'
                    [acc_decorr(cvi), roc_decorr(cvi), MI_decorr(cvi)] = ld_.runLinearDiscriminant(x_trn_decorr,y_trn,x_test_decorr, y_test,params);
                    [sacc_decorr(cvi), sroc_decorr(cvi), sMI_decorr(cvi)] = ld_.runLinearDiscriminant(x_trn_decorr,y_trn_shuffled,x_test_decorr, y_test,params);
            end
        end

    end
    acc_boot(booti) = mean(acc);
    roc_boot(booti) = mean(roc);
    MI_boot(booti) = mean(MI);
    w_boot(:,booti) = mean(w,2);
    bias_boot(booti) = mean(bias);

    sacc_boot(booti) = mean(sacc);
    sroc_boot(booti) = mean(sroc);
    sMI_boot(booti) = mean(sMI);
    sw_boot(:,booti) = mean(sw,2);
    sbias_boot(booti) = mean(sbias);

    if param.decorr
        acc_decorr_boot(booti) = mean(acc_decorr);
        roc_decorr_boot(booti) = mean(roc_decorr);
        MI_decorr_boot(booti) = mean(MI_decorr);
        w_decorr_boot(:,booti) = mean(w_decorr,2);
        bias_decorr_boot(booti) = mean(bias_decorr);

        sacc_decorr_boot(booti) = mean(sacc_decorr);
        sroc_decorr_boot(booti) = mean(sroc_decorr);
        sMI_decorr_boot(booti) = mean(sMI_decorr);
        sw_decorr_boot(:,booti) = mean(sw_decorr,2);
        sbias_decorr_boot(booti) = mean(sbias_decorr);
    end

    disp(['Boot ',num2str(booti),'   time:',num2str(toc),'s']);
end

%% Plot

figure();
tiledlayout(2,3,"TileSpacing","compact","Padding","compact");
nexttile;
errorbar([mean(acc_boot) mean(sacc_boot)],[std(acc_boot) std(sacc_boot)],'Marker','o','LineWidth',2,'Color',[.2 .2 .2],'LineStyle','none');
xlim([0.5 2.5]);
helper.figModule;
ylabel('Accuracy');
nexttile;
errorbar([mean(roc_boot) mean(sroc_boot)],[std(roc_boot) std(sroc_boot)],'Marker','o','LineWidth',2,'Color',[.2 .2 .2],'LineStyle','none');
xlim([0.5 2.5]);
helper.figModule;
ylabel('AUC');
nexttile;
errorbar([mean(MI_boot) mean(sMI_boot)],[std(MI_boot) std(sMI_boot)],'Marker','o','LineWidth',2,'Color',[.2 .2 .2],'LineStyle','none');
xlim([0.5 2.5]);
helper.figModule;
ylabel('MI');

nexttile;
errorbar([mean(acc_decorr_boot) mean(sacc_decorr_boot)],[std(acc_decorr_boot) std(sacc_decorr_boot)],'Marker','o','LineWidth',2,'Color',[.2 .2 .2],'LineStyle','none');
xlim([0.5 2.5]);
helper.figModule;
xticklabels({'Emprical','Shuffled'});
ylabel('Accuracy');
nexttile;
errorbar([mean(roc_decorr_boot) mean(sroc_decorr_boot)],[std(roc_decorr_boot) std(sroc_decorr_boot)],'Marker','o','LineWidth',2,'Color',[.2 .2 .2],'LineStyle','none');
xlim([0.5 2.5]);
helper.figModule;
xticklabels({'Emprical','Shuffled'});
ylabel('AUC');
nexttile;
errorbar([mean(MI_decorr_boot) mean(sMI_decorr_boot)],[std(MI_decorr_boot) std(sMI_decorr_boot)],'Marker','o','LineWidth',2,'Color',[.2 .2 .2],'LineStyle','none');
xlim([0.5 2.5]);
helper.figModule;
xticklabels({'Emprical','Shuffled'});
ylabel('MI');


