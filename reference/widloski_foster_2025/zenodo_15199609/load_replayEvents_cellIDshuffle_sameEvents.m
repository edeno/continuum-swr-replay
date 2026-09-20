clear all

replays_meanPosteriorSpread = [];
replays_meanJumpSize = [];
replays_pvalue = [];

cluster_size = [];

return_fullPosterior = 0;
numShuffles = 100;

Rat_Names = {'Billy3','Curly2','Goethe2'};
for rat = 1:3
cd(strcat(Rat_Names{rat},'/processed'))
if rat == 1 %Billy3
    dayFiles = {'20181205','20181206','20181207','20181208','20181212'};%,'20181218'};
elseif rat == 2 %Curly2
    dayFiles = {'20191007','20191008','20191009','20191010','20191011','20191013'};
elseif rat == 3 %Goethe2
    dayFiles = {'20180607','20180608','20180609','20180614'};
end
%iterate through days
for day = 1:length(dayFiles)
    display(strcat(Rat_Names{rat},' Day ',num2str(day)))
    cd(dayFiles{day})

    load Experiment_Information
    load Analysis_Information
    load clusters
    load replayEvents

    windowSizeDecoding = 0.08;
    shiftSizeDecoding = 0.005;

    for sessionNum_decoder = 1:size(Run_Times,1)
        sessionNum = sessionNum_decoder;

        %select clusters to use for decoding
            [ind_cluster,rateMap_smoothed,~,rateMap_smoothed_NaN,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,ind_cluster_interneurons,ind_cluster_all] = ...
                load_rateMapsForDecoding_2D(clusters,sessionNum_decoder,Run_Times,numSpatialBins,meanFiringRateThr,spatialInfoThr,noiseOverlapThr,isolationThr,peakSnrThr,numSpatialBins_coarse);
            
            rateMap_original = rateMap_smoothed;

            % cluster_size = [cluster_size; length(ind_cluster_all) length(ind_cluster)];
            % continue

        %load replays
            replay_dispersionThr = 12; 
            load_ind_replayEvents

            replays_meanPosteriorSpread_sub = nan(length(ind_replay),numShuffles+1);
            replays_meanJumpSize_sub = nan(length(ind_replay),numShuffles+1);
            replays_pvalue_sub = nan(length(ind_replay),2);
            
            for i = 1:length(ind_replay)
                timeBins = decoder_replay(sessionNum_decoder).replayEvents(ind_replay(i)).timeBins;
                numSpks = load_numSpks_timeBins(timeBins,clusters,ind_cluster,shiftSizeDecoding,windowSizeDecoding);

                numTimeBins = size(numSpks,2);
                numClusters = size(numSpks,1);

                %decoding
                for j = 1:numShuffles+1
                    if j==1
                        ind_rand = (1:length(ind_cluster));
                    else
                        ind_rand = randperm(length(ind_cluster));
                    end
                    rateMap_sub = rateMap_original(ind_rand,:);

                    [~,posteriorCOM,posteriorSpread] =  compute_BayesianDecoding_fast(numSpks,rateMap_sub,numSpatialBins,spatialDim,windowSizeDecoding,return_fullPosterior);
                    jumps = compute_sequenceJumps(posteriorCOM); jumps = [jumps;jumps(end)]; jumps(jumps==0) = nan;

                    replays_meanPosteriorSpread_sub(i,j) = nanmean(posteriorSpread);
                    replays_meanJumpSize_sub(i,j) = nanmean(jumps);
                end

                replays_pvalue_sub(i,1) = (1+nansum(replays_meanPosteriorSpread_sub(i,2:end)<replays_meanPosteriorSpread_sub(i,1)))/(1+numShuffles);
                replays_pvalue_sub(i,2) = (1+nansum(replays_meanJumpSize_sub(i,2:end)<replays_meanJumpSize_sub(i,1)))/(1+numShuffles);

                % subplot(121), histogram(meanPosteriorSpread(i,2:end),100), vline(meanPosteriorSpread(i,1))
                % subplot(122), histogram(meanJumpSize(i,2:end),100), vline(meanJumpSize(i,1))
                % keyboard
            end

        %save
            replays_meanPosteriorSpread = [replays_meanPosteriorSpread; replays_meanPosteriorSpread_sub repmat([rat day sessionNum ind_replay(i)],size(replays_meanPosteriorSpread_sub,1),1)];
            replays_meanJumpSize = [replays_meanJumpSize; replays_meanJumpSize_sub repmat([rat day sessionNum ind_replay(i)],size(replays_meanJumpSize_sub,1),1)];
            replays_pvalue = [replays_pvalue; replays_pvalue_sub];


    end
    cd ..
end
cd ../..
cd analysis_RippleField/
    replays_pvalue = Pvalue;
    replays_meanPosteriorSpread = MeanPosteriorSpread;
    replays_meanJumpSize = MeanJumpSize;
    save('replayEvents_cellIDshuffle_sameEvents.mat','replays_pvalue','replays_meanPosteriorSpread','replays_meanJumpSize')
cd ..
end
keyboard

    cd analysis_RippleField/
        load replayEvents_cellIDshuffle_sameEvents
        load replayEvents_eventStats_rippleThr2sd_wShiftComparison_075secPostReplay.mat

        binSize = 2;
        replays_meanPosteriorSpread = replays_meanPosteriorSpread/binSize;
    cd ..

        pvalueThr = 0.05;
    
        numDetectors = 64;
        A = replays_tets_rand(:,:,:,1); B = replays_tets_rand(:,:,:,2); A(B>pvalueThr) = nan;
        % probSucess = nanmean(nanmean(nansum(A>0,1)./sum(~isnan(replays_tets(:,1)))));
        probSucess = nanmean(nanmean(nansum(A(replays_stats(:,end-4)==1,:,:)>0,1)./sum(~isnan(replays_stats(:,1)) & replays_stats(:,end-4)==1)));
        numTrials = size(replays_tets,1);
        expectedNumDetectors = probSucess*numDetectors;
    
        x = 0:20;
        y = binopdf(x,numDetectors,probSucess);
        numTetsThr = InterX([x; cumsum(y)/sum(y)]',[min(x) max(x); 0.95 0.95]'); numTetsThr = round(numTetsThr(1));


        replays_stats_nanRemoved = replays_stats; replays_stats_nanRemoved(isnan(replays_stats(:,1)),:) = [];
        ind_sessions = unique(replays_stats_nanRemoved(:,end-2:end),'rows');
        data_fractions = nan(size(ind_sessions,1),4,3); 
        data_replayStats_frac = nan(size(ind_sessions,1),1,2,2);
        data_MeanPosteriorSpread = nan(size(ind_sessions,1),11,2,2);
        data_MeanJumpSize = nan(size(ind_sessions,1),11,2,2);
        for i = 1:size(ind_sessions,1)
            % ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows'));
            ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows') & replays_stats(:,end-4)==1);
            replays_stats_sub = replays_stats(ind_session,:);
                replays_stats_sub(:,5:6) = abs(replays_stats_sub(:,5:6)); %take absolute value of the angles
            replays_tets_sub = replays_tets(ind_session,:,:);

            MeanPosteriorSpread_sub = replays_meanPosteriorSpread(ind_session,:);
            MeanJumpSize_sub = replays_meanJumpSize(ind_session,:);

            for k = 1:2
                if k==1
                    %filter for avg tet ripples
                    ind_neither = replays_stats_sub(:,9)==0 & replays_stats_sub(:,10)==0;
                    ind_either = (replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                    ind_both = (replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                    ind_ripp = (replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) & replays_stats_sub(:,10)==0;
                    ind_burst = replays_stats_sub(:,9)==0 & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                    data_fractions(i,:,k) = [sum(ind_neither) sum(ind_both) sum(ind_ripp) sum(ind_burst)]/size(replays_stats_sub,1);

                else
                    %filter for avg tet and individual tet thresholded ripples
                    A = replays_tets_sub(:,:,2); B = replays_tets_sub(:,:,3); A(B>=pvalueThr) = nan;
                    ind_neither = replays_stats_sub(:,9)==0 & nansum(A>0,2)<numTetsThr & replays_stats_sub(:,10)==0;
                    ind_either = (replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr | (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                    ind_both = ((replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                    ind_ripp = ((replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr) & replays_stats_sub(:,10)==0;
                    ind_burst = (replays_stats_sub(:,9)==0 & nansum(A>0,2)<numTetsThr) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                    data_fractions(i,:,k) = [sum(ind_neither) sum(ind_both) sum(ind_ripp) sum(ind_burst)]/size(replays_stats_sub,1);
                end

                data_replayStats_frac(i,1,1,k) = sum(ind_neither)/size(replays_stats_sub,1);
                data_replayStats_frac(i,1,2,k) = sum(ind_either)/size(replays_stats_sub,1);
                data_MeanPosteriorSpread(i,1:end,1,k) = mean(MeanPosteriorSpread_sub(ind_neither,1:11),1,'omitnan');
                data_MeanPosteriorSpread(i,1:end,2,k) = mean(MeanPosteriorSpread_sub(ind_either,1:11),1,'omitnan');
                data_MeanJumpSize(i,1:end,1,k) = mean(MeanJumpSize_sub(ind_neither,1:11),1,'omitnan');
                data_MeanJumpSize(i,1:end,2,k) = mean(MeanJumpSize_sub(ind_either,1:11),1,'omitnan');
            end
        end

        for j = 1:2
            if j==1
                data_1 = data_MeanPosteriorSpread(:,1,1,1);
                data_shuffle_1 = data_MeanPosteriorSpread(:,2:end,1,1);
                data_2 = data_MeanPosteriorSpread(:,1,1,2);
                data_shuffle_2 = data_MeanPosteriorSpread(:,2:end,1,2);
            else
                data_1 = data_MeanJumpSize(:,1,1,1);
                data_shuffle_1 = data_MeanJumpSize(:,2:end,1,1);
                data_2 = data_MeanJumpSize(:,1,1,2);
                data_shuffle_2 = data_MeanJumpSize(:,2:end,1,2);
            end

            subplot(1,2,j)
            data_sub = {data_1,data_2,data_shuffle_1(:),data_shuffle_2(:)};
            pvalues = nan(1,2);
            [~,pvalues(1)] = ttest2(data_sub{1},data_sub{3});
            [~,pvalues(2)] = ttest2(data_sub{2},data_sub{4});
            violin(data_sub,'facecolor',[0 0 1; 1 0 0; 0 1 1; 1 0 1]);
            set(gca,'xtick',1:length(data_sub),'xticklabel',{'w/o ripple and burst','w/ ripple or burst','w/o ripple and burst - SHUFFLE','w/ ripple or burst - SHUFFLE'},'fontsize',14), set(gcf,'color','w')
            sigstar({[1 3],[2 4]},pvalues)
            xlim([0 length(data_sub)+1]), axis square, xtickangle(30)

            if j ==1, ylabel('posterior spread (cm)'), 
            else, ylabel('jump size (cm)'), 
            end
            ax = gca; % Get current axes
            ax.TickLength = [0.02, 0.02];

            pbaspect([2 1 1])
        end

