clear all

windowSizeDecoding = 0.08;
shiftSizeDecoding = 0.005;

numShuffles = 100;
numShuffles_replay = 100;
Duration = cell(3,6,3,numShuffles+1);
Dispersion = cell(3,6,3,numShuffles+1);
Timepoints = cell(3,6,3,numShuffles+1);

cd analysis_RippleField/
    load replayEvents_cellIDshuffle_newEvents
cd ..

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
    load behavior

    %sliding window size
        name_ext = split(string(windowSizeDecoding),'.'); name_ext = name_ext(2);
    
    %time bins
        timeBins = load_timeBins(Times_day,shiftSizeDecoding,windowSizeDecoding);

    for sessionNum_decoder = 1:size(Run_Times,1)
        sessionNum = sessionNum_decoder;

        %decoding time bins for the session
            timeBins_session = compute_dataTemporalConcatenation(timeBins,load_timeBounds(Run_Times{sessionNum_decoder}));
            times_session = nanmean(timeBins_session,2);
            posteriorCOM = nan(size(timeBins_session,1),2);
            posteriorSpread = nan(size(timeBins_session,1),1);

        %rat position
            positions_session = compute_dataTemporalConcatenation(positions,load_timeBounds(Run_Times{sessionNum}));
            positions_session = compute_dataInterpolation(positions_session,times_session);
            ind_good = find(positions_session(:,4)<speedThr);
        
        %select clusters to use for decoding
            [ind_cluster,rateMap_smoothed,~,rateMap_smoothed_NaN,~,~,~,~,~,~,~,~,rateMap_coarse_smoothed,~,~,~,~,~,~,~,~,~,~,~,~,~] = ...
                load_rateMapsForDecoding_2D(clusters,sessionNum_decoder,Run_Times,numSpatialBins,meanFiringRateThr,spatialInfoThr,noiseOverlapThr,isolationThr,peakSnrThr,numSpatialBins_coarse);
            
            rateMap_original = rateMap_smoothed;
            
        %collect spikes within timeBin windows
            numSpks = load_numSpks_timeBins(timeBins_session,clusters,ind_cluster,shiftSizeDecoding,windowSizeDecoding);
            numSpks = numSpks(:,ind_good);

        %decoding
            numReplays = nan(1,numShuffles+1);
            for j = 1:numShuffles+1
                if j==1
                    ind_rand = (1:length(ind_cluster));
                else
                    ind_rand = randperm(length(ind_cluster));
                end
                rateMap = rateMap_original(ind_rand,:);

                return_fullPosterior = 0;
                [~,posteriorCOM_sub,posteriorSpread_sub] =  compute_BayesianDecoding_fast(numSpks,rateMap,numSpatialBins,spatialDim,windowSizeDecoding,return_fullPosterior);
                
                posteriorCOM(ind_good,:) = posteriorCOM_sub;
                posteriorSpread(ind_good) = posteriorSpread_sub;
    
                jumps = compute_sequenceJumps(posteriorCOM); jumps = [jumps;jumps(end)];
                ind = find(posteriorSpread<(sequence_posteriorSpreadThr/binSize) & jumps<(sequence_jumpThr/binSize));
        
                x = [times_session posteriorCOM posteriorSpread jumps]; 
                x_NaN = [x(:,1) nan(size(x(:,2:end)))];
                x_NaN(ind,:) = x(ind,:);
    
                %find sequences bounded by NaNs
                    [boundaries,lengths] = compute_allSequences_NaNseparated(x_NaN(:,2));        
        
                %merge sequences if gap is delx<delxThr and delt<deltThr 
                    [boundaries,lengths] = compute_allSequences_NaNseparated_merge(x_NaN,boundaries,sequence_delxThr,sequence_deltThr,binSize);
                    
                %remove short sequences (< sequence_durationThr)
                    ind_remove = find(lengths<sequence_durationThr/shiftSizeDecoding);
                    boundaries(ind_remove,:) = [];
                    lengths(ind_remove,:) = [];
    
                %loop through candidate replayEvents and compute duration and dispersion
                    duration_sub = nan(size(boundaries,1),1);
                    dispersion_sub = nan(size(boundaries,1),1);
                    timepoints_sub = nan(size(boundaries,1),2);
                    timeBins_sub = cell(size(boundaries,1),1);
                    for i = 1:size(boundaries,1)
                        indSub = boundaries(i,1):boundaries(i,2);
                        replay_NaN = x_NaN(indSub,2:3);
                        duration_sub(i) = x(indSub(end),1)-x(indSub(1),1);
                        dispersion_sub(i) = compute_sequenceDispersion(replay_NaN);
                        timepoints_sub(i,:) = [x(indSub(1),1) x(indSub(end),1)];
                        timeBins_sub{i} = timeBins_session(indSub,:);
                    end

                %find subset of good replays
                    ind_dispersion = (dispersion_sub>(replay_dispersionThr/binSize));
                    ind_duration = (duration_sub>replay_durationThr);
                    ind_replay = find(ind_duration & ind_dispersion);

                %compute pvalues
                    pvalue_posteriorSpread_sub = nan(size(boundaries,1),1);
                    pvalue_jumpSize_sub = nan(size(boundaries,1),1);
                    for i = 1:length(ind_replay)
                        timeBins_replay = timeBins_sub{i};
                        numSpks_replay = load_numSpks_timeBins(timeBins_replay,clusters,ind_cluster,shiftSizeDecoding,windowSizeDecoding);
                
                        meanPosteriorSpread_sub = nan(1,numShuffles_replay+1);
                        meanJumpSize_sub = nan(1,numShuffles_replay+1);
                        for jj = 1:numShuffles_replay+1
                            if jj==1
                                ind_rand = (1:length(ind_cluster));
                            else
                                ind_rand = randperm(length(ind_cluster));
                            end
                            rateMap_sub = rateMap(ind_rand,:);
        
                            [~,posteriorCOM_sub,posteriorSpread_sub] =  compute_BayesianDecoding_fast(numSpks_replay,rateMap_sub,numSpatialBins,spatialDim,windowSizeDecoding,return_fullPosterior);
                            jumps = compute_sequenceJumps(pvalue_jumpSize_sub); jumps = [jumps;jumps(end)]; jumps(jumps==0) = nan;
        
                            meanPosteriorSpread_sub(jj) = nanmean(posteriorSpread_sub);
                            meanJumpSize_sub(jj) = nanmean(jumps);
                        end
                        pvalue_posteriorSpread_sub(ind_replay(i)) = (1+nansum(meanPosteriorSpread_sub(2:end)<meanPosteriorSpread_sub(1)))/(1+numShuffles_replay);
                        pvalue_jumpSize_sub(ind_replay(i)) = (1+nansum(meanJumpSize_sub(2:end)<meanJumpSize_sub(1)))/(1+numShuffles_replay);
                    end

                    ind_pvalue_posteriorSpread = (pvalue_posteriorSpread_sub<=0.05);
                    ind_pvalue_jumpSize = (pvalue_jumpSize_sub<=0.05);
                    ind_replay = find(ind_duration & ind_dispersion & ind_pvalue_posteriorSpread & ind_pvalue_jumpSize);

            %save
                Duration{rat,day,sessionNum,j} = duration_sub(ind_replay);
                Dispersion{rat,day,sessionNum,j} = dispersion_sub(ind_replay);
                Timepoints{rat,day,sessionNum,j} = timepoints_sub(ind_replay,:);

                numReplays(j) = length(ind_replay);
            end
            numReplays

    end
    cd ..
end
cd ../..
cd analysis_RippleField/
    save('replayEvents_cellIDshuffle_newEvents_pvalues.mat','Timepoints','Duration','Dispersion')
cd ..
end

keyboard

cd analysis_RippleField/
    load replayEvents_cellIDshuffle_newEvents
cd ..
numShuffles = 10;
H_duration = [];
H_dispersion = [];
H_duration_shuffle = [];
H_dispersion_shuffle = [];
edges_duration = linspace(0.1,1,40); centers_duration = edges_duration(1:end-1) + mean(diff(edges_duration))/2;
edges_dispersion = linspace(0,16,40); centers_dispersion = edges_dispersion(1:end-1) + mean(diff(edges_dispersion))/2;
% numReplays_rand = [];
% numReplays_true = [];
for rat = 1:3
    for day = 1:6
        for sessionNum = 1:3
            if isempty(Duration{rat,day,sessionNum}), continue, end
            for j = 1:2
                if j==1
                    h_duration = histcounts(Duration{rat,day,sessionNum,j},edges_duration);%,'Normalization','probability');
                    h_dispersion = histcounts(Dispersion{rat,day,sessionNum,j},edges_dispersion);%,'Normalization','probability');

                    H_duration = [H_duration; h_duration];
                    H_dispersion = [H_dispersion; h_dispersion];
                else
                    Duration_sub = [];
                    Dispersion_sub = [];
                    for jj = 1:numShuffles
                        Duration_sub = [Duration_sub; Duration{rat,day,sessionNum,jj+1}];
                        Dispersion_sub = [Dispersion_sub; Dispersion{rat,day,sessionNum,jj+1}];

                        % numReplays_rand_sub = length(find(Duration{rat,day,sessionNum,jj+1}>0.4 & Dispersion{rat,day,sessionNum,jj+1}>10));
                        % numReplays_rand = [numReplays_rand; numReplays_rand_sub];
                    end
                    h_duration_shuffle = histcounts(Duration_sub,edges_duration)/numShuffles;%,'Normalization','probability');
                    h_dispersion_shuffle = histcounts(Dispersion_sub,edges_dispersion)/numShuffles;%,'Normalization','probability'); 
                
                    H_duration_shuffle = [H_duration_shuffle; h_duration_shuffle];
                    H_dispersion_shuffle = [H_dispersion_shuffle; h_dispersion_shuffle];

                    % ind_session = find(ismember(numReplays(:,2:end),[rat day sessionNum],'rows'));
                    % numReplays_true = [numReplays_true; numReplays(ind_session,1)];
                    
                end
            end
        end
    end
end

% edges = linspace(0,5,10); centers = edges(1:end-1) + mean(diff(edges));
% h = histcounts(numReplays_rand,linspace(0,5,10))/numShuffles;
% plot(centers,h)
% hold on, histogram(numReplays_true,edges), hold off

set(gcf,'color','w')
subplot(231)
plot(centers_dispersion,nanmean(H_dispersion),'r','linewidth',2)
hold on, plot(centers_dispersion,nanmean(H_dispersion_shuffle),'b','linewidth',2), hold off
ylabel('num. of events'), xlabel('dispersion'), axis square, set(gca,'fontsize',14), set(gca,'yscale','linear')
legend('data','shuffle')

subplot(232)
plot(centers_dispersion,nanmean(H_dispersion),'r','linewidth',2)
hold on, plot(centers_dispersion,nanmean(H_dispersion_shuffle),'b','linewidth',2), hold off
ylabel('num. of events'), xlabel('dispersion'), axis square, set(gca,'fontsize',14), set(gca,'yscale','linear')
ylim([0 8])
vline(10,'g')

subplot(233)
plot(centers_dispersion,nanmean(H_dispersion),'r','linewidth',2)
hold on, plot(centers_dispersion,nanmean(H_dispersion_shuffle),'b','linewidth',2), hold off
ylabel('num. of events'), xlabel('dispersion'), axis square, set(gca,'fontsize',14), set(gca,'yscale','log')

subplot(234)
plot(centers_duration,nanmean(H_duration),'r','linewidth',2)
hold on, plot(centers_duration,nanmean(H_duration_shuffle),'b','linewidth',2), hold off
ylabel('num. of events'), xlabel('duration (sec)'), axis square, set(gca,'fontsize',14), set(gca,'yscale','linear')
legend('data','shuffle')

subplot(235)
plot(centers_duration,nanmean(H_duration),'r','linewidth',2)
hold on, plot(centers_duration,nanmean(H_duration_shuffle),'b','linewidth',2), hold off
ylabel('num. of events'), xlabel('duration (sec)'), axis square, set(gca,'fontsize',14), set(gca,'yscale','linear')
ylim([0 40])
vline(0.5,'g')

subplot(236)
plot(centers_duration,nanmean(H_duration),'r','linewidth',2)
hold on, plot(centers_duration,nanmean(H_duration_shuffle),'b','linewidth',2), hold off
ylabel('num. of events'), xlabel('duration (sec)'), axis square, set(gca,'fontsize',14), set(gca,'yscale','log')
