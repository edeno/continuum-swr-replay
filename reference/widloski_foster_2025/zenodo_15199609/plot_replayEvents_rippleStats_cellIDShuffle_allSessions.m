clear all

cd analysis_RippleField/
    load replayEvents_cellIDshuffle_newEvents
    numCellIDShuffles = size(Duration,4);
cd ..

replays_stats_cell = cell(1,numCellIDShuffles);
replays_tets_cell = cell(1,numCellIDShuffles);
replays_tets_rand_cell = cell(1,numCellIDShuffles);

ii = 1;
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
    
    %load data
        load Experiment_Information
        load Analysis_Information
        load behavior
        load clusters
        load mazeTimes
        load binDecoding_08
        load spikeDensity, spikeDensity_full = spikeDensity;
            
    %event detection
        eventMergeThr = 0;%0.05;
        eventZscoreThr_rippleEvent = 2; eventDurationThr_rippleEvent = 0.015;
        eventZscoreThr_spikeDensityEvent = 3; eventDurationThr_spikeDensityEvent = 0.05;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
    %iterate through sessions
    for sessionNum_decoder = 1:length(Run_Times)
        sessionNum = sessionNum_decoder;
        
            %times
                Times = load_timeBounds(Run_Times{sessionNum});  
                trialTimes = load_fieldVec([session(sessionNum).trials],'trialTimes',2);
                lickingTimes = load_fieldVec([session(sessionNum).trials],'lickingTimes',2);
                drinkingTimes = load_fieldVec([session(sessionNum).trials],'drinkingTimes',2);
                runningTimes = [trialTimes(diff(trialTimes,[],2)<30,1) lickingTimes(diff(trialTimes,[],2)<30,1)];

                positions_session = compute_dataTemporalConcatenation(positions,load_timeBounds(Run_Times{sessionNum}));
                x = positions_session(:,4); x(positions_session(:,4)>speedThr) = nan;
                [boundaries,lengths] = compute_allSequences_NaNseparated(x);        
                immobilityTimes = [positions_session(boundaries(:,1),1) positions_session(boundaries(:,2),1)];

            %load rate maps for decoding
                [ind_cluster,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,ind_cluster_interneurons,ind_cluster_all] = load_rateMapsForDecoding_2D(clusters,sessionNum_decoder,Run_Times,numSpatialBins,meanFiringRateThr,spatialInfoThr,noiseOverlapThr,isolationThr,peakSnrThr,numSpatialBins_coarse);

            %load LFP
                name = strcat('LFP_',num2str(sessionNum),'.mat');
                if exist(name,'file')==0, continue, end
                load(name)

            %ripple power
                ripplePower = LFP_ripplePower_mean;
                ripplePower_drinkingOnly = compute_dataTemporalConcatenation(ripplePower,drinkingTimes);
                ripplePower_immobility = compute_dataTemporalConcatenation(ripplePower,immobilityTimes);

                    mean_ripplePower = mean(ripplePower_immobility(:,end),'omitnan');
                    std_ripplePower = std(ripplePower_immobility(:,end),'omitnan');

                ripplePower(:,3) = (ripplePower(:,2)-mean_ripplePower)/std_ripplePower;
                ripplePower_drinkingOnly(:,3) = (ripplePower_drinkingOnly(:,2)-mean_ripplePower)/std_ripplePower;
                ripplePower_immobility(:,3) = (ripplePower_immobility(:,2)-mean_ripplePower)/std_ripplePower;

            %ripple power during immobility periods across tetrodes
                times_LFP_drinkingOnly = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),drinkingTimes);
                times_LFP_immobility = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),immobilityTimes);
                    if length(times_LFP_immobility) ~= size(LFP_immobility_ripplePower,2)
                        immobilityTimes(diff(immobilityTimes,[],2)<1,:) = []; 
                        times_LFP_immobility = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),immobilityTimes);
                    end

                ripplePower_tets_drinkingOnly = [times_LFP_drinkingOnly(:,1) LFP_drinking_ripplePower'];
                ripplePower_tets_immobility = [times_LFP_immobility(:,1) LFP_immobility_ripplePower'];

                    mean_ripplePower_tets = mean(ripplePower_tets_immobility(:,2:end),'omitnan');
                    std_ripplePower_tets = std(ripplePower_tets_immobility(:,2:end),'omitnan');

                ripplePower_tets_drinkingOnly(:,2:end) = (ripplePower_tets_drinkingOnly(:,2:end)-repmat(mean_ripplePower_tets,size(ripplePower_drinkingOnly,1),1))./repmat(std_ripplePower_tets,size(ripplePower_tets_drinkingOnly,1),1);
                ripplePower_tets_immobility(:,2:end) = (ripplePower_tets_immobility(:,2:end)-repmat(mean_ripplePower_tets,size(ripplePower_tets_immobility,1),1))./repmat(std_ripplePower_tets,size(ripplePower_tets_immobility,1),1);             

            %Spike Density
                spikeDensity = compute_dataTemporalConcatenation(load_spikeDensity_placeCells(clusters,ind_cluster_all,spikeDensityStepSize),Times); 
                spikeDensity = [spikeDensity(:,1:2) conv(spikeDensity(:,2),setUp_gaussFilt([1 1000],0.1/spikeDensityStepSize),'same')];
                spikeDensity_drinkingOnly = compute_dataTemporalConcatenation(spikeDensity,drinkingTimes);
                spikeDensity_immobility = compute_dataTemporalConcatenation(spikeDensity,immobilityTimes);

                    mean_spikeDensity = mean(spikeDensity_immobility(:,end),'omitnan');
                    std_spikeDensity = std(spikeDensity_immobility(:,end),'omitnan');

                spikeDensity(:,4) = (spikeDensity(:,3)-mean_spikeDensity)/std_spikeDensity;
                spikeDensity_drinkingOnly(:,4) = (spikeDensity_drinkingOnly(:,3)-mean_spikeDensity)/std_spikeDensity;
                spikeDensity_immobility(:,4) = (spikeDensity_immobility(:,3)-mean_spikeDensity)/std_spikeDensity;
                spikeDensity = compute_dataInterpolation(spikeDensity,ripplePower(:,1),[]);
                spikeDensity_drinkingOnly = compute_dataInterpolation(spikeDensity_drinkingOnly,ripplePower(:,1),[]);
                spikeDensity_immobility = compute_dataInterpolation(spikeDensity_immobility,ripplePower(:,1),[]);

            %concatenate data
                Data = [ripplePower(:,1) ripplePower(:,end) spikeDensity(:,end)];
                Data_drinkingOnly = compute_dataTemporalConcatenation(Data,drinkingTimes);

            %remove LFP during drinking periods associated with ripples and replay
                rippleEvents_drinkingOnly = compute_events_times([ripplePower_drinkingOnly(:,1),ripplePower_drinkingOnly(:,end)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent);
                spikeDensityEvents_drinkingOnly = compute_events_times([spikeDensity_drinkingOnly(:,1),spikeDensity_drinkingOnly(:,end)],eventZscoreThr_spikeDensityEvent,eventMergeThr,eventDurationThr_spikeDensityEvent);

        %load replays
            for jj = 1:numCellIDShuffles
                
                Duration_sub = Duration{rat,day,sessionNum,jj};
                Dispersion_sub = Dispersion{rat,day,sessionNum,jj};
                Timepoints_sub = Timepoints{rat,day,sessionNum,jj};
    
                replay_durationThr = 0.1;%0.2,0.4;
                replay_dispersionThr = 12;%16,20;
                replayTimes = Timepoints_sub(Duration_sub>replay_durationThr & Dispersion_sub>replay_dispersionThr/binSize,:);
                numReplays = size(replayTimes,1);

                    [~,ind_replayTimes] = compute_dataTemporalConcatenation(times_LFP_drinkingOnly,[replayTimes; rippleEvents_drinkingOnly(:,1:2); spikeDensityEvents_drinkingOnly(:,1:2)]); ind_replayTimes = find(~isnan(ind_replayTimes(:,2)));
                    ripplePower_tets_drinkingOnly_noReplay = ripplePower_tets_drinkingOnly; ripplePower_tets_drinkingOnly_noReplay(ind_replayTimes,:) = [];
                    Data_drinkingOnly_noReplay = Data_drinkingOnly; Data_drinkingOnly_noReplay(ind_replayTimes,:) = [];
                
                if isempty(replayTimes), continue, end

                Data_replayOnly = compute_dataTemporalConcatenation(Data,replayTimes);
                ripplePower_tets_replayOnly = compute_dataTemporalConcatenation(ripplePower_tets_immobility,replayTimes);
                
                if isempty(ripplePower_tets_replayOnly), continue, end

                %shuffles
                numShuffles = 100;
                replays_stats_sub = nan(numReplays,19);
                replays_tets_sub = nan(numReplays,64,4);
                replays_tets_rand_sub = nan(numReplays,64,numShuffles,2);
                for i = 1:numReplays
                    
                    timePoint = replayTimes(i,:);
                    duration = diff(timePoint);
                    meanReplaySpeed = nan;
                    ratSpeed = nan;
                    elapsedTime = mean(timePoint) - Run_Times{sessionNum}(1,1);
                    angularDisplacement_past = nan;
                    angularDisplacement_future = nan;
                    trialType = nan;
                    lickingEvent = nan;
                    drinkingEvent = sum(compute_isNumWithinRange(mean(timePoint),drinkingTimes));
                    timeBeforeMoving = nan; timeSinceStopping = nan;
                                               
                    Data_replay = compute_dataTemporalConcatenation(Data_replayOnly,timePoint);
                    if isempty(Data_replay), continue, end
    
                    %peak ripple power and spike density
                        ripplePowerPeak = max(Data_replay(:,2));
                        spikeDensityPeak = max(Data_replay(:,3));

                    %peak ripple power across tetrodes
                        ripplePower_tets_replay = compute_dataTemporalConcatenation(ripplePower_tets_replayOnly,timePoint);
                        ripplePowerPeak_tets = max(ripplePower_tets_replay(:,2:end),[],1);
                        if isempty(ripplePower_tets_replay), continue, end
    
                    %number of ripple and burst events
                        numRippleEvents = size(compute_events_times([Data_replay(:,1),Data_replay(:,2)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                        numSpikeDensityEvents = size(compute_events_times([Data_replay(:,1),Data_replay(:,3)],eventZscoreThr_spikeDensityEvent,eventMergeThr,eventDurationThr_spikeDensityEvent),1);
                        numRippleEvents_tets = nan(1,64);
                        for j = 1:64
                            numRippleEvents_tets(j) = size(compute_events_times([ripplePower_tets_replay(:,1),ripplePower_tets_replay(:,j+1)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                        end

                    % %peak ripple power across tetrodes during rand periods
                    %     randShifts = sort(randi(size(ripplePower_tets_immobility,1)-size(Data_replay,1),numShuffles,1));
                    %     ripplePowerPeak_tets_rand = nan(numShuffles,64);
                    %     numRippleEvents_tets_rand = nan(numShuffles,64);
                    %     for j = 1:size(randShifts,1)
                    %         ripplePower_tets_replay_rand_sub = ripplePower_tets_immobility(randShifts(j):randShifts(j) + size(Data_replay,1),:);
                    %         if max(diff(ripplePower_tets_replay_rand_sub(:,1)))>0.1, continue, end
                    %         ripplePowerPeak_tets_rand(j,:) = max(ripplePower_tets_replay_rand_sub(:,2:end),[],1);
                    % 
                    %         for k = 1:64
                    %             if sum(~isnan(ripplePower_tets_replay_rand_sub(:,k+1)))==0, continue, end
                    %             numRippleEvents_tets_rand(j,k) = size(compute_events_times([ripplePower_tets_replay_rand_sub(:,1),ripplePower_tets_replay_rand_sub(:,k+1)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                    %         end
                    %     end
    
                    %shuffles and pvalues
                        randShifts = randi(size(Data_drinkingOnly_noReplay,1)-size(Data_replay,1),numShuffles,1);
                        ripplePowerPeak_shuffle = nan(1,numShuffles);
                        spikeDensityPeak_shuffle = nan(1,numShuffles); 
                        numRippleEvents_shuffle = nan(1,numShuffles);
                        numSpikeDensityEvents_shuffle = nan(1,numShuffles);
                        ripplePowerPeak_tets_shuffle = nan(64,numShuffles);
                        numRippleEvents_tets_shuffle = nan(64,numShuffles);
                        for j = 1:numShuffles
                            Data_shuffle = Data_drinkingOnly_noReplay(randShifts(j):randShifts(j)+size(Data_replay,1),:);
                            if max(diff(Data_shuffle(:,1)))>0.1, continue, end
                            ripplePowerPeak_shuffle(j) = max(Data_shuffle(:,2));
                            spikeDensityPeak_shuffle(j) = max(Data_shuffle(:,3));
                            numRippleEvents_shuffle(j) = size(compute_events_times([Data_shuffle(:,1),Data_shuffle(:,2)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                            numSpikeDensityEvents_shuffle(j) = size(compute_events_times([Data_shuffle(:,1),Data_shuffle(:,3)],eventZscoreThr_spikeDensityEvent,eventMergeThr,eventDurationThr_spikeDensityEvent),1);
    
                            ripplePower_tets_drinkingOnly_noReplay_shuffle = ripplePower_tets_drinkingOnly_noReplay(randShifts(j):randShifts(j)+size(Data_replay,1),:);
                            ripplePowerPeak_tets_shuffle(:,j) = max(ripplePower_tets_drinkingOnly_noReplay_shuffle(:,2:end));
                            for k = 1:64
                                numRippleEvents_tets_shuffle(k,j) = size(compute_events_times([ripplePower_tets_drinkingOnly_noReplay_shuffle(:,1),ripplePower_tets_drinkingOnly_noReplay_shuffle(:,k+1)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                            end
                        end
                        ripplePowerPeak_pvalue = (1+nansum(ripplePowerPeak_shuffle>ripplePowerPeak))/(1+sum(~isnan(ripplePowerPeak_shuffle)));
                        spikeDensityPeak_pvalue = (1+nansum(spikeDensityPeak_shuffle>spikeDensityPeak))/(1+sum(~isnan(spikeDensityPeak_shuffle)));
                        numRippleEvents_pvalue = (1+nansum(numRippleEvents_shuffle>=numRippleEvents))/(1+sum(~isnan(numRippleEvents_shuffle)));
                        numSpikeDensityEvents_pvalue = (1+nansum(numSpikeDensityEvents_shuffle>=numSpikeDensityEvents))/(1+sum(~isnan(numSpikeDensityEvents_shuffle)));
    
                        ripplePowerPeak_tets_pvalue = nan(1,64);
                        numRippleEvents_tets_pvalue = nan(1,64);
                        for j = 1:64
                            ripplePowerPeak_tets_pvalue(j) = (1+nansum(ripplePowerPeak_tets_shuffle(j,:)>=ripplePowerPeak_tets(j)))/(1+sum(~isnan(ripplePowerPeak_tets_shuffle(j,:))));
                            numRippleEvents_tets_pvalue(j) = (1+nansum(numRippleEvents_tets_shuffle(j,:)>=numRippleEvents_tets(j)))/(1+sum(~isnan(numRippleEvents_tets_shuffle(j,:))));
                        end
    
                        % ripplePowerPeak_tets_rand_pvalue = nan(numShuffles,64);
                        % for j = 1:64
                        %     for k = 1:numShuffles
                        % %         ripplePowerPeak_tets_rand_pvalue(k,j) = (1+nansum(ripplePowerPeak_tets_shuffle(j,:)>=ripplePowerPeak_tets_rand(k,j)))/(1+sum(~isnan(ripplePowerPeak_tets_shuffle(j,:))));
                        %     end
                        % end
                        
                    %save
                        replays_stats_sub(i,:) = [elapsedTime timeSinceStopping timeBeforeMoving ratSpeed angularDisplacement_past angularDisplacement_future duration meanReplaySpeed...
                                                 numRippleEvents numSpikeDensityEvents ripplePowerPeak_pvalue spikeDensityPeak_pvalue...
                                                 trialType lickingEvent drinkingEvent i rat day sessionNum];
    
                        replays_tets_sub(i,:,1) = ripplePowerPeak_tets;
                        replays_tets_sub(i,:,2) = numRippleEvents_tets;
                        replays_tets_sub(i,:,3) = ripplePowerPeak_tets_pvalue;
                        replays_tets_sub(i,:,4) = numRippleEvents_tets_pvalue;
    
                        % replays_tets_rand_sub(i,:,:,1) = numRippleEvents_tets_rand';
                        % replays_tets_rand_sub(i,:,:,2) = ripplePowerPeak_tets_rand_pvalue';
       
                end
                replays_stats_cell{jj} = [replays_stats_cell{jj}; replays_stats_sub];
                replays_tets_cell{jj} = [replays_tets_cell{jj}; replays_tets_sub];
                % replays_tets_rand_cell{jj} = [replays_tets_rand_cell{jj}; replays_tets_rand_sub];
            end

        ii = ii+1;
    end
    cd ..
end
cd ../..
cd analysis_RippleField/
save('replayEvents_eventStats_cellIDShuffle_2.mat','replays_stats_cell','replays_tets_cell','replays_tets_rand_cell','-v7.3')
cd ..
end

keyboard

cd analysis_RippleField/
    load replayEvents_eventStats_cellIDShuffle
    % load replayEvents_eventStats_cellIDShuffle_highThreshold
    % load replayEvents_eventStats_cellIDShuffle_medThreshold
cd ..

%1           2                 3                4        5                        6                          7        8                                                 
%elapsedTime timeSinceStopping timeBeforeMoving ratSpeed angularDisplacement_past angularDisplacement_future duration meanReplaySpeed 

%9               10                    11                     12                  
%numRippleEvents numSpikeDensityEvents ripplePowerPeak_pvalue spikeDensityPeak_pvalue 

%13        14           15            16       17  18  19
%trialType lickingEvent drinkingEvent replayID rat day sessionNum


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = nan(37,101);
for jj = 1:101
    replays_stats = replays_stats_cell{jj};
    replays_tets = replays_tets_cell{jj};
    replays_tets_rand = replays_tets_rand_cell{jj};

    if isempty(replays_stats), continue, end

    pvalueThr = 0.05;

    % numDetectors = 64;
    % A = replays_tets_rand(:,:,:,1); B = replays_tets_rand(:,:,:,2); A(B>pvalueThr) = nan;
    % probSucess = nanmean(nanmean(nansum(A(replays_stats(:,end-4)==1,:,:)>0,1)./sum(~isnan(replays_stats(:,1)) & replays_stats(:,end-4)==1)));
    %     if isnan(probSucess), continue, end
    % numTrials = size(replays_tets,1);
    % expectedNumDetectors = probSucess*numDetectors;
    % x = 0:20;
    % y = binopdf(x,numDetectors,probSucess);
    % numTetsThr = InterX([x; cumsum(y)/sum(y)]',[min(x) max(x); 0.95 0.95]'); numTetsThr = round(numTetsThr(1));
    numTetsThr = 15;

    replays_stats_nanRemoved = replays_stats; replays_stats_nanRemoved(isnan(replays_stats(:,1)),:) = [];
    ind_sessions = unique(replays_stats_nanRemoved(:,end-2:end),'rows');
    data_fractions = nan(size(ind_sessions,1),4); 
    for i = 1:size(ind_sessions,1)
        % ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows'));
        ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows') & replays_stats(:,end-4)==1);
        % ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows') & replays_stats(:,end-4)==1 & replays_pvalue(:,1)<=0.05 & replays_pvalue(:,2)<=0.05);
        replays_stats_sub = replays_stats(ind_session,:);
        replays_tets_sub = replays_tets(ind_session,:,:);

        %filter for avg tet and individual tet thresholded ripples
        A = replays_tets_sub(:,:,2); B = replays_tets_sub(:,:,3); A(B>=pvalueThr) = nan;
        ind_neither = replays_stats_sub(:,9)==0 & nansum(A>0,2)<numTetsThr & replays_stats_sub(:,10)==0;
        ind_either = (replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr | (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
        ind_both = ((replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
        ind_ripp = ((replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr) & replays_stats_sub(:,10)==0;
        ind_burst = (replays_stats_sub(:,9)==0 & nansum(A>0,2)<numTetsThr) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
        data_fractions(i,:) = [sum(ind_neither) sum(ind_both) sum(ind_ripp) sum(ind_burst)];%/size(replays_stats_sub,1);

        data(i,jj) = sum(ind_neither);%/size(replays_stats_sub,1);
    end
end

pvalues = nan(1,37);
for i = 1:37
    pvalues(i) = (1+nansum(data(i,2:end)>data(i,1)))/(1+sum(~isnan(data(i,2:end))));
end
[X2, p_combined] = fishers_combined_method(pvalues);

set(gcf,'color','w')
subplot(121)
k = 3;
edges = 0:1:25; centers = edges(1:end-1) + mean(diff(edges));
% edges = 0:1:8; centers = edges(1:end-1) + mean(diff(edges));
h = histcounts(data(k,2:end),edges);
plot(centers,h,'k','linewidth',2)
vline(data(k,1),'k:')
xlabel('num. of ripple/burst-less replays'), ylabel('count')
axis square, set(gca,'fontsize',12)
xl = xlim; yl = ylim;
text(data(3,1)+diff(xl)*0.05,yl(end)-diff(yl)*0.1,strcat('p=',num2str(compute_round(pvalues(k),100))),'HorizontalAlignment','left','fontsize',12)
xlim([min(edges) max(edges)])

subplot(122)
histogram(pvalues,linspace(0,1,20),'EdgeColor','none','FaceColor','k')
vline(0.05,'k--')
xlabel('p-value'), ylabel('num. of sessions')
axis square, set(gca,'fontsize',12)
xl = xlim; yl = ylim;
text(0.05+diff(xl)*0.05,yl(end)-diff(yl)*0.1,strcat(num2str(sum(pvalues<=0.05)),'/',num2str(length(pvalues)),load_pstar(p_combined)),'HorizontalAlignment','left','fontsize',12)
