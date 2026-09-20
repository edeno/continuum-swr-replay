clear all

replays_stats = [];
replays_powerSpectrum = [];
replays_powerSpectrum_shuffle = [];
replays_tets = [];
replays_numRippleEvents_tets_shuffle = [];
replays_ripplePowerPeak_tets_shuffle = [];
replays_stats_shift = [];
replays_powerSpectrum_shift = [];
replays_tets_shift = [];
numRippleEvents_shift_vec = [];
replays_tets_rand = [];
data_tetrodePositions = [];

shiftPeriod = 0.75;%0.5;
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
        load replayEvents
        load clusters
        load mazeTimes
        load binDecoding_08
        load spikeDensity, spikeDensity_full = spikeDensity;

        cd .., load tetrodePositions, cd(dayFiles{day})
            
        %event detection
            eventMergeThr = 0;%0.05;

            eventZscoreThr_rippleEvent = 2; eventDurationThr_rippleEvent = 0.015;
            % eventZscoreThr_rippleEvent = 3; eventDurationThr_rippleEvent = 0.05;
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
                % immobilityTimes(diff(immobilityTimes,[],2)<1,:) = [];

            %load replays
                replay_dispersionThr = 12;
                replay_durationThr = 0.1;
                load_ind_replayEvents
                replayTimes = timePoints(ind_replay,:);
                    replayTimes_old = timePoints(ind_replay_old,:);
                replayShiftTimes = [replayTimes(:,2) replayTimes(:,2)+4];
                    replayShiftTimes_old = [replayTimes_old(:,2) replayTimes_old(:,2)+4];

            %load rate maps for decoding
                [ind_cluster,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,ind_cluster_interneurons,ind_cluster_all] = load_rateMapsForDecoding_2D(clusters,sessionNum_decoder,Run_Times,numSpatialBins,meanFiringRateThr,spatialInfoThr,noiseOverlapThr,isolationThr,peakSnrThr,numSpatialBins_coarse);

            %load LFP
                name = strcat('LFP_',num2str(sessionNum),'.mat');
                if exist(name,'file')==0, continue, end
                load(name)

                name = strcat('LFP_',num2str(sessionNum),'_replayShift_raw.mat');
                load(name)

            %find times with spurious (non-ripple) deflections in the raw LFP
                times_remove = [];
                for i = 1:10
                    [pks,locs] = findpeaks(abs(diff(LFP_raw(:,i+1)))); 
                    locs(pks<500) = [];
                    times_remove = [LFP_raw(locs,1)-1 LFP_raw(locs,1)+1];
                end

            %raw LFP during drinking across tetrodes
                times_LFP_replayOnly = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),replayTimes_old);
                times_LFP_drinkingOnly = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),drinkingTimes);
                times_LFP_immobility = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),immobilityTimes);
                    if length(times_LFP_immobility) ~= size(LFP_immobility_ripplePower,2)
                        immobilityTimes(diff(immobilityTimes,[],2)<1,:) = []; 
                        times_LFP_immobility = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),immobilityTimes);
                    end
                times_LFP_replayShift = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),replayShiftTimes_old);
                
                LFP_raw_drinkingOnly = [times_LFP_drinkingOnly(:,1) LFP_drinking_raw'];
                % LFP_raw_immobility = [times_LFP_immobility(:,1) LFP_immobility_raw'];
                LFP_raw_replayOnly = [times_LFP_replayOnly(:,1) LFP_replay_raw'];
                    LFP_raw_replayOnly = compute_dataTemporalConcatenation(LFP_raw_replayOnly,replayTimes);
                LFP_raw_replayShift = [times_LFP_replayShift(:,1) LFP_replayShift_raw'];
                    LFP_raw_replayShift = compute_dataTemporalConcatenation(LFP_raw_replayShift,replayShiftTimes);

            %ripple power
                ripplePower = LFP_ripplePower_mean;
                ripplePower_replayOnly = compute_dataTemporalConcatenation(LFP_ripplePower_mean,replayTimes);
                ripplePower_drinkingOnly = compute_dataTemporalConcatenation(ripplePower,drinkingTimes);
                ripplePower_immobility = compute_dataTemporalConcatenation(ripplePower,immobilityTimes);

                %compute mean and standard deviation
                    % %method 1
                    % bins = linspace(0,50,200); centers = bins(1:end-1) + mean(diff(bins))/2;
                    % h = histcounts(ripplePower_immobility(:,end),bins,'normalization','probability');
                    % [~,ind_mode] = maxk(h,5); ind_mode = round(nanmean(ind_mode)); 
                    % h_reflect = [h(1:ind_mode),h(ind_mode:-1:1)];
                    % center_mode = nanmean(centers(ind_mode));
                    % 
                    % L1 = [centers(1:length(h_reflect)); cumsum(h_reflect)/sum(h_reflect)]';
                    % L2 = [0 centers(length(h_reflect)); 0.99 0.99]';
                    % thr = InterX(L1,L2); thr = thr(1);
                    % 
                    % mean_ripplePower = center_mode;
                    % std_ripplePower = thr - center_mode;

                %         % plot(centers,h,'k','linewidth',3)
                %         % hold on, plot(centers(1:length(h_reflect)),h_reflect,'r:','linewidth',3), hold off
                %         % vline(center_mode,'k--'), xlim([min(centers) max(centers)])
                %         % vline(thr,'g-'), xlim([min(centers) max(centers)])
                %         % xlabel('ripple power'), ylabel('fraction'), set(gca,'fontsize',14), set(gcf,'color','w'), axis square
                %         % legend('data','shuffle')

                    %method 2
                    mean_ripplePower = mean(ripplePower_immobility(:,end),'omitnan');
                    std_ripplePower = std(ripplePower_immobility(:,end),'omitnan');

                    % %method 3
                    % mean_ripplePower = mean(ripplePower_drinkingOnly(:,end),'omitnan');
                    % std_ripplePower = std(ripplePower_drinkingOnly(:,end),'omitnan');

                ripplePower(:,3) = (ripplePower(:,2)-mean_ripplePower)/std_ripplePower;
                ripplePower_replayOnly(:,3) = (ripplePower_replayOnly(:,2)-mean_ripplePower)/std_ripplePower;
                ripplePower_drinkingOnly(:,3) = (ripplePower_drinkingOnly(:,2)-mean_ripplePower)/std_ripplePower;
                ripplePower_immobility(:,3) = (ripplePower_immobility(:,2)-mean_ripplePower)/std_ripplePower;

            %ripple power during drinking periods across tetrodes
                ripplePower_tets_replayOnly = [times_LFP_replayOnly(:,1) LFP_replay_ripplePower'];
                    ripplePower_tets_replayOnly = compute_dataTemporalConcatenation(ripplePower_tets_replayOnly,replayTimes);
                ripplePower_tets_drinkingOnly = [times_LFP_drinkingOnly(:,1) LFP_drinking_ripplePower'];
                ripplePower_tets_immobility = [times_LFP_immobility(:,1) LFP_immobility_ripplePower'];

                %compute mean and standard deviation
                    % %method 1
                    % mean_ripplePower_tets = nan(1,64);
                    % std_ripplePower_tets = nan(1,64);
                    % bins = linspace(-20,80,200); centers = bins(1:end-1) + mean(diff(bins))/2;
                    % for i = 1:64
                    %     if sum(~isnan(ripplePower_tets_immobility(:,i+1)) & ripplePower_tets_immobility(:,i+1)~=0)==0, continue, end
                    %     h = histcounts(ripplePower_tets_immobility(:,i+1),bins,'normalization','probability');
                    %     [~,ind_mode] = maxk(h,5); ind_mode = round(nanmean(ind_mode)); 
                    %     h_reflect = [h(1:ind_mode),h(ind_mode:-1:1)];
                    %     center_mode = nanmean(centers(ind_mode));
                    % 
                    %     if sum(h_reflect)==0, continue, end
                    %     L1 = [centers(1:length(h_reflect)); cumsum(h_reflect)/sum(h_reflect)]';
                    %     L2 = [0 centers(length(h_reflect)); 0.99 0.99]';
                    %     thr = InterX(L1,L2); thr = thr(1);
                    % 
                    %     mean_ripplePower_tets(i) = center_mode;
                    %     std_ripplePower_tets(i) = thr - center_mode;
                    % 
                    %         % plot(centers,h,'k','linewidth',3)
                    %         % hold on, plot(centers(1:length(h_reflect)),h_reflect,'r:','linewidth',3), hold off
                    %         % vline(center_mode,'k--'), xlim([min(centers) max(centers)])
                    %         % vline(thr,'g-'), xlim([min(centers) max(centers)])
                    %         % xlabel('ripple power'), ylabel('fraction'), set(gca,'fontsize',14), set(gcf,'color','w'), axis square
                    %         % legend('data','shuffle')
                    %         % keyboard
                    % end

                    %method 2
                        mean_ripplePower_tets = mean(ripplePower_tets_immobility(:,2:end),'omitnan');
                        std_ripplePower_tets = std(ripplePower_tets_immobility(:,2:end),'omitnan');

                    % %method 3
                    %     mean_ripplePower_tets = mean(ripplePower_tets_drinkingOnly(:,2:end),'omitnan');
                    %     std_ripplePower_tets = std(ripplePower_tets_drinkingOnly(:,2:end),'omitnan');
                
                ripplePower_tets_replayOnly(:,2:end) = (ripplePower_tets_replayOnly(:,2:end)-repmat(mean_ripplePower_tets,size(ripplePower_replayOnly,1),1))./repmat(std_ripplePower_tets,size(ripplePower_tets_replayOnly,1),1);
                ripplePower_tets_drinkingOnly(:,2:end) = (ripplePower_tets_drinkingOnly(:,2:end)-repmat(mean_ripplePower_tets,size(ripplePower_drinkingOnly,1),1))./repmat(std_ripplePower_tets,size(ripplePower_tets_drinkingOnly,1),1);
                ripplePower_tets_immobility(:,2:end) = (ripplePower_tets_immobility(:,2:end)-repmat(mean_ripplePower_tets,size(ripplePower_tets_immobility,1),1))./repmat(std_ripplePower_tets,size(ripplePower_tets_immobility,1),1);

                
                threshold = 5;
                rippleEvents_replayOnly = compute_events_times([ripplePower_replayOnly(:,1),ripplePower_replayOnly(:,end)],5,eventMergeThr,eventDurationThr_rippleEvent);
                LFP_tets_replayOnly = [times_LFP_replayOnly(:,1) LFP_replay_raw'];
                pairs = compute_pairs(64);
                ripplePhaseShift_tets = nan(size(rippleEvents_replayOnly,1),size(pairs,1));
                for i = 1:size(rippleEvents_replayOnly,1)
                    LFP_tets_replayOnly_sub = compute_dataTemporalConcatenation(LFP_raw_replayOnly,rippleEvents_replayOnly(i,1:2));
                    LFP_tets_replayOnly_sub_extended = compute_dataTemporalConcatenation(LFP_raw_replayOnly,rippleEvents_replayOnly(i,1:2) + [-0.5 0.5]);
                    ripplePower_tets_replayOnly_sub = compute_dataTemporalConcatenation(ripplePower_tets_replayOnly,rippleEvents_replayOnly(i,1:2));
                    if max(diff(LFP_tets_replayOnly_sub(:,1)))>0.1, continue, end

                    for j = 1:size(pairs,1)
                        ripplePower1 = ripplePower_tets_replayOnly_sub(:,1+pairs(j,1));
                        ripplePower2 = ripplePower_tets_replayOnly_sub(:,1+pairs(j,2));
                        if max(ripplePower1)<threshold || max(ripplePower2)<threshold || isnan(max(ripplePower1)) || isnan(max(ripplePower2)), continue, end

                        vec1 = LFP_tets_replayOnly_sub(:,1+pairs(j,1));
                        vec2 = LFP_tets_replayOnly_sub(:,1+pairs(j,2));

                        SWRFreqRange = [100 220];
                        [~,~,vec1] = compute_filteredLFP(SWRFreqRange,vec1,LFPSampRate); 
                        [~,~,vec2] = compute_filteredLFP(SWRFreqRange,vec2,LFPSampRate); 

                        [C,lags] = xcorr(vec1,vec2);
                        [C1,lags] = xcorr(vec1);

                        [~,locs] = findpeaks(C1); if isempty(locs), continue, end
                        locs(lags(locs)==0) = [];
                        lag_min = find(min(abs(lags(locs)))==abs(lags(locs)));
                        period = abs(lags(locs(lag_min(1))));

                        [~,locs] = findpeaks(C); if isempty(locs), continue, end
                        lag_min = find(min(abs(lags(locs)))==abs(lags(locs)));
                        lag = abs(lags(locs(lag_min(1))));

                        phaseShift = lag/period*2*pi;

                        ripplePhaseShift_tets(i,j) = phaseShift;

                        % subplot(211), plot(vec1), hold on, plot(vec2), hold off
                        % subplot(212), plot(lags,C), xlim([-50 50])
                        % vline(lag)
                        % title(phaseShift)
                    end
                end

                ind_sameSide = find((pairs(:,1)<=32 & pairs(:,2)<=32) | (pairs(:,1)>32 & pairs(:,2)>32));
                ind_oppSide = find((pairs(:,1)<=32 & pairs(:,2)>32) | (pairs(:,1)>32 & pairs(:,2)<=32));

                % plot(nanmean(ripplePhaseShift_tets(:,ind_sameSide)'))
                % hold on, plot(nanmean(ripplePhaseShift_tets(:,ind_oppSide)')), hold off

                distance = nan(size(pairs,1),1);
                for i = 1:length(distance)
                    distance(i) = sqrt((tetrodePositions(pairs(i,1),1) - tetrodePositions(pairs(i,2),1)).^2 + (tetrodePositions(pairs(i,1),2) - tetrodePositions(pairs(i,2),2)).^2);
                end
                distance(ind_oppSide) = nan;

                data_tetrodePositions = [data_tetrodePositions,{[distance ripplePhaseShift_tets' repmat([rat day sessionNum],length(distance),1)]}];

                % edges = linspace(0,2,10); centers = [edges(1:end-1) + mean(diff(edges))/2 2.2];
                % data = nan(2,length(edges));
                % for i = 1:length(edges)
                %     if i<length(edges)
                %         ind = find(distance>edges(i) & distance<=edges(i+1));
                %     else 
                %         ind = find(isnan(distance));
                %     end
                %     if isempty(ind), continue, end
                %     data(1,i) = nanmean(nanmean(abs(ripplePhaseShift_tets(:,ind))));
                %     data(2,i) = nanstd(nanmean(abs(ripplePhaseShift_tets(:,ind))))./sqrt(length(ind));
                % end
                % shadedErrorBar(centers,data(1,:),data(2,:))

            %theta power
                thetaPower = LFP_thetaPower_mean;
                thetaPower_immobility = compute_dataTemporalConcatenation(thetaPower,immobilityTimes);
                thetaPower_drinkingOnly = compute_dataTemporalConcatenation(thetaPower,immobilityTimes);
                    mean_thetaPower = mean(thetaPower_immobility(:,end),'omitnan');
                    std_thetaPower = std(thetaPower_immobility(:,end),'omitnan');
                thetaPower(:,3) = (thetaPower(:,2)-mean_thetaPower)/std_thetaPower;

            %posterior spread
                sequence_posteriorSpreadThr = inf;
                sequence_spikeDensityThr = -inf;
                sequence_jumpThr = inf;
                [x_immobility_bins,~,~,posteriorSpread,~,~,~,posteriorPeak,jumpSize,timeBins_NaNremoved_merge] = compute_filtering_binDecoding(decoder_binDecoding(sessionNum),sequence_posteriorSpreadThr,[0 speedThr],sequence_spikeDensityThr,positions,spikeDensity_full,spikeDensityStepSize,sequence_jumpThr,load_timeBounds(Run_Times{sessionNum}),binSize);
                posteriorSpread = compute_dataInterpolation([x_immobility_bins(:,1) posteriorSpread],ripplePower(:,1),[]);
                jumpSize = compute_dataInterpolation([x_immobility_bins(:,1) jumpSize],ripplePower(:,1),[]);

            %Spike Density
                % spikeDensity = compute_dataTemporalConcatenation(load_spikeDensity_placeCells(clusters,1:length(clusters),spikeDensityStepSize),Times); 
                spikeDensity = compute_dataTemporalConcatenation(load_spikeDensity_placeCells(clusters,ind_cluster_all,spikeDensityStepSize),Times); 
                % spikeDensity = compute_dataTemporalConcatenation(load_spikeDensity_placeCells(clusters,ind_cluster,spikeDensityStepSize),Times); 
                spikeDensity_smoothed = conv(spikeDensity(:,2),setUp_gaussFilt([1 1000],0.1/spikeDensityStepSize),'same');
                spikeDensity = [spikeDensity(:,1) spikeDensity(:,2) spikeDensity_smoothed];
                spikeDensity_drinkingOnly = compute_dataTemporalConcatenation(spikeDensity,drinkingTimes);
                spikeDensity_immobility = compute_dataTemporalConcatenation(spikeDensity,immobilityTimes);

                    mean_spikeDensity = mean(spikeDensity_immobility(:,end),'omitnan');
                    std_spikeDensity = std(spikeDensity_immobility(:,end),'omitnan');

                    % mean_spikeDensity = mean(spikeDensity_drinkingOnly(:,end),'omitnan');
                    % std_spikeDensity = std(spikeDensity_drinkingOnly(:,end),'omitnan');

                spikeDensity(:,4) = (spikeDensity(:,3)-mean_spikeDensity)/std_spikeDensity;
                spikeDensity_drinkingOnly(:,4) = (spikeDensity_drinkingOnly(:,3)-mean_spikeDensity)/std_spikeDensity;
                spikeDensity_immobility(:,4) = (spikeDensity_immobility(:,3)-mean_spikeDensity)/std_spikeDensity;
                spikeDensity = compute_dataInterpolation(spikeDensity,ripplePower(:,1),[]);
                spikeDensity_drinkingOnly = compute_dataInterpolation(spikeDensity_drinkingOnly,ripplePower(:,1),[]);
                spikeDensity_immobility = compute_dataInterpolation(spikeDensity_immobility,ripplePower(:,1),[]);

                % %plot
                %     spikeDensity_smoothed_short = spikeDensity_smoothed(1:10000);
                %     [P,F,T] = compute_powerSpectrogram(spikeDensity_smoothed_short,1/spikeDensityStepSize);
                %     hp = pcolor(T,F,P); hp.EdgeColor = 'none';
                %     ylim([0 60])

            Data = [ripplePower(:,1) ripplePower(:,end) spikeDensity(:,end) thetaPower(:,end) posteriorSpread(:,end)/binSize jumpSize(:,end)/binSize];
            Data_drinkingOnly = compute_dataTemporalConcatenation(Data,drinkingTimes);

            %remove LFP during drinking periods associated with ripples and replay
                rippleEvents_drinkingOnly = compute_events_times([ripplePower_drinkingOnly(:,1),ripplePower_drinkingOnly(:,end)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent);
                spikeDensityEvents_drinkingOnly = compute_events_times([spikeDensity_drinkingOnly(:,1),spikeDensity_drinkingOnly(:,end)],eventZscoreThr_spikeDensityEvent,eventMergeThr,eventDurationThr_spikeDensityEvent);

                [~,ind_replayTimes] = compute_dataTemporalConcatenation(LFP_raw_drinkingOnly,[replayTimes; rippleEvents_drinkingOnly(:,1:2); spikeDensityEvents_drinkingOnly(:,1:2)]); ind_replayTimes = find(~isnan(ind_replayTimes(:,2)));
                % [~,ind_replayTimes] = compute_dataTemporalConcatenation(LFP_raw_drinkingOnly,replayTimes); ind_replayTimes = find(~isnan(ind_replayTimes(:,2)));
                LFP_raw_drinkingOnly_noReplay = LFP_raw_drinkingOnly; LFP_raw_drinkingOnly_noReplay(ind_replayTimes,:) = [];
                ripplePower_tets_drinkingOnly_noReplay = ripplePower_tets_drinkingOnly; ripplePower_tets_drinkingOnly_noReplay(ind_replayTimes,:) = [];
                Data_drinkingOnly_noReplay = Data_drinkingOnly; Data_drinkingOnly_noReplay(ind_replayTimes,:) = [];
                ripplePower_immobility_noReplay = ripplePower_immobility; ripplePower_immobility_noReplay(ind_replayTimes,:) = [];
                
                    numShuffles = 100;
                    shiftPeriod_vec = 0:0.1:4;
                    numRippleEvents_shift_vec_sub = zeros(size(replayTimes,1),length(shiftPeriod_vec));
                    for i = 1:length(shiftPeriod_vec)
                        [~,ripplePower_immobility_replay_shift] = compute_dataTemporalConcatenation(ripplePower,replayTimes + shiftPeriod_vec(i));
                        [boundaries,lengths] = compute_allSequences_NaNseparated(ripplePower_immobility_replay_shift(:,2));   
                        for j = 1:size(boundaries,1)
                            events = compute_events_times([ripplePower_immobility_replay_shift(boundaries(j,1):boundaries(j,2),1),ripplePower_immobility_replay_shift(boundaries(j,1):boundaries(j,2),end)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent);
                            ripplePowerPeak_shift = max(ripplePower_immobility_replay_shift(boundaries(j,1):boundaries(j,2),end));
                            if size(events,1)>0
                                randShifts = randi(size(ripplePower_immobility_noReplay,1)-lengths(j),numShuffles,1);
                                ripplePowerPeak_shift_shuffle = nan(1,numShuffles);
                                for k = 1:numShuffles
                                    ripplePowerPeak_shift_shuffle(k) = max(ripplePower_immobility_noReplay(randShifts(k):randShifts(k)+lengths(j),end));
                                end
                                ripplePowerPeak_pvalue = (1+nansum(ripplePowerPeak_shift_shuffle>ripplePowerPeak_shift))/(1+sum(~isnan(ripplePowerPeak_shift_shuffle)));
                                numRippleEvents_shift_vec_sub(j,i) = ripplePowerPeak_pvalue<0.05;
                            end
                        end
                    end
                    numRippleEvents_shift_vec = [numRippleEvents_shift_vec; nanmean(numRippleEvents_shift_vec_sub)];
                    % plot(shiftPeriod_vec,nanmean(numRippleEvents_shift_vec,1),'b','linewidth',3)
                    % axis square, set(gca,'fontsize',14), xlabel('time post replay (sec)'), ylabel('')
                    % drawnow
                    % continue

            numShuffles = 100;
            replays_stats_sub = nan(length(ind_replay),46);
            replays_powerSpectrum_sub = nan(length(ind_replay),1000);
            replays_powerSpectrum_shift_sub = nan(length(ind_replay),1000);
            replays_powerSpectrum_shuffle_sub = nan(length(ind_replay),1000);
            replays_tets_sub = nan(length(ind_replay),64,4);
            replays_tets_shift_sub = nan(length(ind_replay),64,4);
            replays_numRippleEvents_tets_shuffle_sub = nan(length(ind_replay),64,numShuffles);
            replays_ripplePowerPeak_tets_shuffle_sub = nan(length(ind_replay),64,numShuffles);
            replays_tets_rand_sub = nan(length(ind_replay),64,numShuffles);
            for i = 1:length(ind_replay)
                
                timePoints = decoder_replay(sessionNum).replayEvents(ind_replay(i)).timePoints;
                timePoints_shift = timePoints + diff(timePoints) + shiftPeriod;
                replay = decoder_replay(sessionNum).replayEvents(ind_replay(i)).replay;
                indNaN = decoder_replay(sessionNum).replayEvents(ind_replay(i)).indNaN;
                replay_NaN = replay; replay_NaN(indNaN,:) = nan;
                jumps = [nan; compute_sequenceJumps(replay_NaN)];
                meanReplaySpeed = binSize*mean(jumps,'omitnan')./shiftSizeDecoding;
                ratSpeed = decoder_replay(sessionNum).replayEvents(ind_replay(i)).ratSpeed;
                posteriorSpreadMean = decoder_replay(sessionNum).replayEvents(ind_replay(i)).posteriorSpreadMean;
                elapsedTime = mean(timePoints) - Run_Times{sessionNum}(1,1);
                angularDisplacement_past = decoder_replay(sessionNum).replayEvents(ind_replay(i)).meanAngDisplacement_pastPath;
                angularDisplacement_future = decoder_replay(sessionNum).replayEvents(ind_replay(i)).meanAngDisplacement_futPath;
                trialType = decoder_replay(sessionNum).replayEvents(ind_replay(i)).trialType;
                lickingEvent = decoder_replay(sessionNum).replayEvents(ind_replay(i)).lickingEvent;
                drinkingEvent = sum(compute_isNumWithinRange(mean(timePoints),drinkingTimes));
                timeBeforeMoving = nan; timeSinceStopping = nan;
                ind_timeBeforeMoving = find(diff(sign(mean(timePoints) - trialTimes(:,1)))~=0)+1;
                ind_timeSinceStopping = find(diff(sign(mean(timePoints) - lickingTimes(:,1)))~=0);
                if ~isempty(ind_timeBeforeMoving), timeBeforeMoving = trialTimes(ind_timeBeforeMoving,1) - mean(timePoints); end
                if ~isempty(ind_timeSinceStopping), timeSinceStopping = mean(timePoints) - lickingTimes(ind_timeSinceStopping,1); end
                                                    
                Data_replay = compute_dataTemporalConcatenation(Data,timePoints);
                Data_replay_shift = compute_dataTemporalConcatenation(Data,timePoints_shift);
                if isempty(Data_replay), continue, end

                %max ripple power across tetrodes
                    ripplePower_tets_replay = compute_dataTemporalConcatenation(ripplePower_tets_replayOnly,timePoints);
                    ripplePowerPeak_tets = max(ripplePower_tets_replay(:,2:end),[],1);

                %shift period
                    ripplePower_tets_replay_shift = compute_dataTemporalConcatenation(ripplePower_tets_immobility,timePoints_shift);
                    ripplePowerPeak_tets_shift = max(ripplePower_tets_replay_shift(:,2:end),[],1);
                    if isempty(ripplePowerPeak_tets_shift), continue, end

                %rand period %!!!Change ripplePower_tets_immobility to ripplePower_tets_drinkingOnly if only using licking periods for final analysis
                    randShifts = sort(randi(size(ripplePower_tets_immobility,1)-size(Data_replay,1),numShuffles,1));
                    ripplePowerPeak_tets_rand = nan(numShuffles,64);
                    numRippleEvents_tets_rand = nan(numShuffles,64);
                    for j = 1:size(randShifts,1)
                        ripplePower_tets_replay_rand_sub = ripplePower_tets_immobility(randShifts(j):randShifts(j) + size(Data_replay,1),:);
                        if max(diff(ripplePower_tets_replay_rand_sub(:,1)))>0.1, continue, end
                        ripplePowerPeak_tets_rand(j,:) = max(ripplePower_tets_replay_rand_sub(:,2:end),[],1);

                        for k = 1:64
                            if sum(~isnan(ripplePower_tets_replay_rand_sub(:,k+1)))==0, continue, end
                            numRippleEvents_tets_rand(j,k) = size(compute_events_times([ripplePower_tets_replay_rand_sub(:,1),ripplePower_tets_replay_rand_sub(:,k+1)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                        end
                    end

                ripplePowerPeak = max(Data_replay(:,2));
                ripplePowerMean = mean(Data_replay(:,2),'omitnan');
                spikeDensityPeak = max(Data_replay(:,3));
                spikeDensityMean = mean(Data_replay(:,3),'omitnan');
                thetaPowerMean = mean(Data_replay(:,4),'omitnan');
                posteriorSpreadMean = mean(Data_replay(:,5),'omitnan');
                jumpSizeMean = mean(Data_replay(:,6),'omitnan');

                ripplePowerPeak_shift = max(Data_replay_shift(:,2));
                ripplePowerMean_shift = mean(Data_replay_shift(:,2),'omitnan');
                spikeDensityPeak_shift = max(Data_replay_shift(:,3));
                spikeDensityMean_shift = mean(Data_replay_shift(:,3),'omitnan');
                thetaPowerMean_shift = mean(Data_replay_shift(:,4),'omitnan');
                posteriorSpreadMean_shift = mean(Data_replay_shift(:,5),'omitnan');
                jumpSizeMean_shift = mean(Data_replay_shift(:,6),'omitnan');

                spikeVec = nan(length(clusters),1);
                spikeVec_shift = nan(length(clusters),1);
                for j = 1:length(clusters)
                    spkTimes = clusters(j).spkTime;
                    spkTimes(find(diff(spkTimes)==0),:) = [];
                    spikeVec(j) = histcounts(spkTimes,timePoints);
                    spikeVec_shift(j) = histcounts(spkTimes,timePoints_shift);
                end
                numCells = sum(spikeVec>0)/diff(timePoints);
                meanSpks = sum(spikeVec)/diff(timePoints);
                numCells_shift = sum(spikeVec_shift>0)/diff(timePoints_shift);
                meanSpks_shift = sum(spikeVec_shift)/diff(timePoints_shift);

                %number of events
                    numRippleEvents = size(compute_events_times([Data_replay(:,1),Data_replay(:,2)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                    numSpikeDensityEvents = size(compute_events_times([Data_replay(:,1),Data_replay(:,3)],eventZscoreThr_spikeDensityEvent,eventMergeThr,eventDurationThr_spikeDensityEvent),1);
                    numRippleEvents_tets = nan(1,64);
                    for j = 1:64
                        numRippleEvents_tets(j) = size(compute_events_times([ripplePower_tets_replay(:,1),ripplePower_tets_replay(:,j+1)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                    end

                    numRippleEvents_shift = size(compute_events_times([Data_replay_shift(:,1),Data_replay_shift(:,2)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                    numSpikeDensityEvents_shift = size(compute_events_times([Data_replay_shift(:,1),Data_replay_shift(:,3)],eventZscoreThr_spikeDensityEvent,eventMergeThr,eventDurationThr_spikeDensityEvent),1);
                    numRippleEvents_tets_shift = nan(1,64);
                    if size(ripplePower_tets_replay_shift,1)>10
                        for j = 1:64
                            if sum(~isnan(ripplePower_tets_replay_shift(:,j+1)))==0, continue, end
                            numRippleEvents_tets_shift(j) = size(compute_events_times([ripplePower_tets_replay_shift(:,1),ripplePower_tets_replay_shift(:,j+1)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                        end
                    end

                %power spectrum (replay vs random snippet within drinking period)
                    LFP_raw_singleReplay = compute_dataTemporalConcatenation(LFP_raw_replayOnly,timePoints);
                    LFP_raw_singleReplay_shift = compute_dataTemporalConcatenation(LFP_raw_replayShift,timePoints_shift);
                    P = nan(64,1000);
                    P_shift = nan(64,1000);
                    P_shuffle = nan(64,1000,numShuffles);
                    randShifts = randi(size(LFP_raw_drinkingOnly_noReplay,1)-size(Data_replay,1),numShuffles,1);
                    for j = 1:64
                        [P_sub,f] = compute_powerSpectrum(LFP_raw_singleReplay(:,j+1),LFPSampRate); 
                        P(j,:) = P_sub(1:1000);

                        [P_sub,f] = compute_powerSpectrum(LFP_raw_singleReplay_shift(:,j+1),LFPSampRate); 
                        P_shift(j,:) = P_sub(1:1000);

                        for k = 1:numShuffles
                            times_shuffle = LFP_raw_drinkingOnly_noReplay(randShifts(k):randShifts(k)+size(Data_replay,1),1);
                            if max(diff(times_shuffle))>0.1, continue, end
                            [P_sub,f] = compute_powerSpectrum(LFP_raw_drinkingOnly_noReplay(randShifts(k):randShifts(k)+size(Data_replay,1),j+1),LFPSampRate); 
                            P_shuffle(j,:,k) = P_sub(1:1000);
                        end
                    end
                    ind_goodTets = find(nansum(P,2)>0);
                    if rat == 3, ind_goodTets(ismember(ind_goodTets,[61 22 11])) = []; end
                    replays_powerSpectrum_sub(i,:) = nanmean(P(ind_goodTets,:));
                    replays_powerSpectrum_shift_sub(i,:) = nanmean(P_shift(ind_goodTets,:));
                    replays_powerSpectrum_shuffle_sub(i,:) = nanmean(nanmean(P_shuffle(ind_goodTets,:,:),3));


                    % if numRippleEvents==0 && numSpikeDensityEvents==0
                    %     subplot(121)
                    %     plot(f(1:1000),(P(numRippleEvents_tets(ind_goodTets)==0,:)'),'r','linewidth',1)
                    %     hold on, plot(f(1:1000),(P(numRippleEvents_tets(ind_goodTets)>0,:)'),'k','linewidth',1), hold off
                    %     hold on, plot(f(1:1000),(nanmean(P_shuffle(numRippleEvents_tets(ind_goodTets)==0,:,:),3)'),'g','linewidth',1), hold off
                    %     set(gca,'yscale','log')
                    % 
                    %     subplot(122)
                    %     plot(f(1:1000),nanmean(P(numRippleEvents_tets(ind_goodTets)==0,:)),'r','linewidth',2)
                    %     hold on, plot(f(1:1000),nanmean(P(numRippleEvents_tets(ind_goodTets)>0,:)),'k','linewidth',2), hold off
                    %     hold on, plot(f(1:1000),nanmean(nanmean(P_shuffle(numRippleEvents_tets(ind_goodTets)==0,:,:),3)),'g','linewidth',2), hold off
                    %     set(gca,'yscale','log')
                    %     keyboard
                    % end

                    % subplot(311), plot(LFP_raw_replay(:,1),LFP_raw_replay(:,1+ind_goodTets))
                    % subplot(312), 
                    %     plot(ripplePower_tets_replay(:,1),ripplePower_tets_replay(:,1+ind_goodTets)), 
                    %     title([numRippleEvents,max(numRippleEvents_tets)])
                    %     ylim([-1 10])
                    % subplot(313), 
                    %     plot(f(1:1000),P(ind_goodTets,:)), xlim([0 250])
                    %     ylim([0 8])
                    %     vline(ThetaFreqRange,'r:'), vline([100 220],'b:')
                    % keyboard

                    % subplot(131), plot(Data_replay(:,3),'k','linewidth',2)
                    % subplot(132), plot(Data_replay(:,4),'k','linewidth',2)
                    % subplot(133), plot(Data_replay(:,5),'k','linewidth',2)

                %shuffles
                    randShifts = randi(size(Data_drinkingOnly_noReplay,1)-size(Data_replay,1),numShuffles,1);
                    ripplePowerPeak_shuffle = nan(1,numShuffles);
                    ripplePowerMean_shuffle = nan(1,numShuffles);
                    spikeDensityPeak_shuffle = nan(1,numShuffles); 
                    spikeDensityMean_shuffle = nan(1,numShuffles); 
                    thetaPowerMean_shuffle = nan(1,numShuffles); 
                    posteriorSpreadMean_shuffle = nan(1,numShuffles); 
                    jumpSizeMean_shuffle = nan(1,numShuffles); 
                    numRippleEvents_shuffle = nan(1,numShuffles);
                    numSpikeDensityEvents_shuffle = nan(1,numShuffles);
                    ripplePowerPeak_tets_shuffle = nan(64,numShuffles);
                    numRippleEvents_tets_shuffle = nan(64,numShuffles);
                    numCells_shuffle = nan(1,numShuffles);
                    meanSpks_shuffle = nan(1,numShuffles);
                    timeRelativeToMovement_shuffle = nan(2,numShuffles);
                    for j = 1:numShuffles
                        times_shuffle = Data_drinkingOnly_noReplay(randShifts(j):randShifts(j)+size(Data_replay,1),1);
                        if max(diff(times_shuffle))>0.1, continue, end
                        Data_shuffle = Data_drinkingOnly_noReplay(randShifts(j):randShifts(j)+size(Data_replay,1),:);
                        ripplePower_tets_drinkingOnly_noReplay_shuffle = ripplePower_tets_drinkingOnly_noReplay(randShifts(j):randShifts(j)+size(Data_replay,1),:);
                        timePoints_shuffle = [min(times_shuffle) max(times_shuffle)];

                        spikeVec = nan(length(clusters),1);
                        for k = 1:length(clusters)
                            spkTimes = clusters(k).spkTime;
                            spkTimes(find(diff(spkTimes)==0),:) = [];
                            spikeVec(k) = histcounts(spkTimes,timePoints_shuffle);
                        end
                        numCells_shuffle(j) = sum(spikeVec>0)/diff(timePoints_shuffle);
                        meanSpks_shuffle(j) = sum(spikeVec)/diff(timePoints_shuffle);

                        ripplePowerPeak_shuffle(j) = max(Data_shuffle(:,2));
                        ripplePowerMean_shuffle(j) = mean(Data_shuffle(:,2),'omitnan');
                        spikeDensityPeak_shuffle(j) = max(Data_shuffle(:,3));
                        spikeDensityMean_shuffle(j) = mean(Data_shuffle(:,3),'omitnan');
                        thetaPowerMean_shuffle(j) = mean(Data_shuffle(:,4),'omitnan');
                        posteriorSpreadMean_shuffle(j) = mean(Data_shuffle(:,5),'omitnan');
                        jumpSizeMean_shuffle(j) = mean(Data_shuffle(:,6),'omitnan');
                        numRippleEvents_shuffle(j) = size(compute_events_times([Data_shuffle(:,1),Data_shuffle(:,2)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                        numSpikeDensityEvents_shuffle(j) = size(compute_events_times([Data_shuffle(:,1),Data_shuffle(:,3)],eventZscoreThr_spikeDensityEvent,eventMergeThr,eventDurationThr_spikeDensityEvent),1);

                        ripplePowerPeak_tets_shuffle(:,j) = max(ripplePower_tets_drinkingOnly_noReplay_shuffle(:,2:end));
                        for k = 1:64
                            numRippleEvents_tets_shuffle(k,j) = size(compute_events_times([ripplePower_tets_drinkingOnly_noReplay_shuffle(:,1),ripplePower_tets_drinkingOnly_noReplay_shuffle(:,k+1)],eventZscoreThr_rippleEvent,eventMergeThr,eventDurationThr_rippleEvent),1);
                        end

                        % subplot(131), hold on, plot(Data_shuffle(:,3)), hold off
                        % subplot(132), hold on, plot(Data_shuffle(:,4)), hold off
                        % subplot(133), hold on, plot(Data_shuffle(:,5)), hold off

                        % plot(ripplePower_tets_drinkingOnly_noReplay_shuffle(:,1),ripplePower_tets_drinkingOnly_noReplay_shuffle(:,2:end)')
                        % keyboard
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

                    ripplePowerPeak_shift_pvalue = (1+nansum(ripplePowerPeak_shuffle>ripplePowerPeak_shift))/(1+sum(~isnan(ripplePowerPeak_shuffle)));
                    spikeDensityPeak_shift_pvalue = (1+nansum(spikeDensityPeak_shuffle>spikeDensityPeak_shift))/(1+sum(~isnan(spikeDensityPeak_shuffle)));
                    numRippleEvents_shift_pvalue = (1+nansum(numRippleEvents_shuffle>=numRippleEvents_shift))/(1+sum(~isnan(numRippleEvents_shuffle)));
                    numSpikeDensityEvents_shift_pvalue = (1+nansum(numSpikeDensityEvents_shuffle>=numSpikeDensityEvents_shift))/(1+sum(~isnan(numSpikeDensityEvents_shuffle)));

                    ripplePowerPeak_tets_shift_pvalue = nan(1,64);
                    numRippleEvents_tets_shift_pvalue = nan(1,64);
                    for j = 1:64
                        ripplePowerPeak_tets_shift_pvalue(j) = (1+nansum(ripplePowerPeak_tets_shuffle(j,:)>=ripplePowerPeak_tets_shift(j)))/(1+sum(~isnan(ripplePowerPeak_tets_shuffle(j,:))));
                        numRippleEvents_tets_shift_pvalue(j) = (1+nansum(numRippleEvents_tets_shuffle(j,:)>=numRippleEvents_tets_shift(j)))/(1+sum(~isnan(numRippleEvents_tets_shuffle(j,:))));
                    end

                    ripplePowerPeak_tets_rand_pvalue = nan(numShuffles,64);
                    for j = 1:64
                        for k = 1:numShuffles
                            ripplePowerPeak_tets_rand_pvalue(k,j) = (1+nansum(ripplePowerPeak_tets_shuffle(j,:)>=ripplePowerPeak_tets_rand(k,j)))/(1+sum(~isnan(ripplePowerPeak_tets_shuffle(j,:))));
                        end
                    end
                    
                %save
                    replays_stats_sub(i,:) = [elapsedTime timeSinceStopping timeBeforeMoving ratSpeed angularDisplacement_past angularDisplacement_future duration(ind_replay(i)) meanReplaySpeed...
                                             numRippleEvents numSpikeDensityEvents ripplePowerPeak_pvalue spikeDensityPeak_pvalue...
                                             ripplePowerPeak spikeDensityPeak ripplePowerMean spikeDensityMean thetaPowerMean posteriorSpreadMean jumpSizeMean meanSpks numCells...
                                             nanmean(ripplePowerPeak_shuffle) nanmean(spikeDensityPeak_shuffle) nanmean(ripplePowerMean_shuffle) nanmean(spikeDensityMean_shuffle) nanmean(thetaPowerMean_shuffle) nanmean(posteriorSpreadMean_shuffle) nanmean(jumpSizeMean_shuffle) nanmean(meanSpks_shuffle) nanmean(numCells_shuffle)...
                                             ripplePowerPeak_shift spikeDensityPeak_shift ripplePowerMean_shift spikeDensityMean_shift thetaPowerMean_shift posteriorSpreadMean_shift jumpSizeMean_shift meanSpks_shift numCells_shift...
                                             trialType lickingEvent drinkingEvent ind_replay(i) rat day sessionNum];
                    

                    replays_tets_sub(i,:,1) = ripplePowerPeak_tets;
                    replays_tets_sub(i,:,2) = numRippleEvents_tets;
                    replays_tets_sub(i,:,3) = ripplePowerPeak_tets_pvalue;
                    replays_tets_sub(i,:,4) = numRippleEvents_tets_pvalue;

                    replays_tets_shift_sub(i,:,1) = ripplePowerPeak_tets_shift;
                    replays_tets_shift_sub(i,:,2) = numRippleEvents_tets_shift;
                    replays_tets_shift_sub(i,:,3) = ripplePowerPeak_tets_shift_pvalue;
                    replays_tets_shift_sub(i,:,4) = numRippleEvents_tets_shift_pvalue;

                    replays_tets_rand_sub(i,:,:,1) = numRippleEvents_tets_rand';
                    replays_tets_rand_sub(i,:,:,2) = ripplePowerPeak_tets_rand_pvalue';

                    replays_ripplePowerPeak_tets_shuffle_sub(i,:,:) = ripplePowerPeak_tets_shuffle;
                    replays_numRippleEvents_tets_shuffle_sub(i,:,:) = numRippleEvents_tets_shuffle;

            end
            replays_stats = [replays_stats; replays_stats_sub];
            replays_powerSpectrum = [replays_powerSpectrum; replays_powerSpectrum_sub];
            replays_powerSpectrum_shuffle = [replays_powerSpectrum_shuffle; replays_powerSpectrum_shuffle_sub];
            replays_powerSpectrum_shift = [replays_powerSpectrum_shift; replays_powerSpectrum_shift_sub];
            replays_tets = [replays_tets; replays_tets_sub];
            replays_tets_shift = [replays_tets_shift; replays_tets_shift_sub];
            replays_tets_rand = [replays_tets_rand; replays_tets_rand_sub];
            replays_ripplePowerPeak_tets_shuffle = [replays_ripplePowerPeak_tets_shuffle; replays_ripplePowerPeak_tets_shuffle_sub];
            replays_numRippleEvents_tets_shuffle = [replays_numRippleEvents_tets_shuffle; replays_numRippleEvents_tets_shuffle_sub];

        ii = ii+1;
    end
    cd ..
end
cd ../..
cd analysis_RippleField/
save('replayEvents_eventStats_rippleThr2sd_wShiftComparison_075secPostReplay_spkDensityAllClusters_sigReplays.mat','replays_stats','replays_powerSpectrum','replays_powerSpectrum_shuffle','replays_tets','f','replays_ripplePowerPeak_tets_shuffle','replays_numRippleEvents_tets_shuffle','replays_powerSpectrum_shift','replays_tets_shift','replays_tets_rand','numRippleEvents_shift_vec','shiftPeriod_vec')
cd ..
end

keyboard

cd analysis_RippleField/
% load replayEvents_eventStats_rippleThr2sd_wShiftComparison_075secPostReplay.mat
% load replayEvents_eventStats_rippleThr2sd_wShiftComparison_075secPostReplay_spkDensityAllClusters
load replayEvents_eventStats_rippleThr2sd_wShiftComparison_075secPostReplay_spkDensityAllClusters_sigReplays

% load replayEvents_eventStats_rippleThr2sd_wShiftComparison_075secPostReplay_spkDensityPlaceCells
% load replayEvents_eventStats_rippleThr2sd_wShiftComparison_075secPostReplay_spkDensityAllClusters
% load replayEvents_eventStats_rippleThr2sd_wShiftComparison_075secPostReplay_spkDensityAllClusters_highThreshold

load replayEvents_cellIDshuffle_sameEvents
cd ..

%1           2                 3                4        5                        6                          7        8                                                 
%elapsedTime timeSinceStopping timeBeforeMoving ratSpeed angularDisplacement_past angularDisplacement_future duration meanReplaySpeed 

%9               10                    11                     12                  
%numRippleEvents numSpikeDensityEvents ripplePowerPeak_pvalue spikeDensityPeak_pvalue 

%13              14               15              16               17             18                  19           20         21                                       
%ripplePowerPeak spikeDensityPeak ripplePowerMean spikeDensityMean thetaPowerMean posteriorSpreadMean jumpSizeMean meanSpikes numCells 

%22                      23                       24                      25                       26                     27                          28                   29                 30                                   
%ripplePowerPeak_shuffle spikeDensityPeak_shuffle ripplePowerMean_shuffle spikeDensityMean_shuffle thetaPowerMean_shuffle posteriorSpreadMean_shuffle jumpSizeMean_shuffle meanSpikes_shuffle numCells_shuffle

%31                    32                     33                    34                     35                   36                        37                 38               39                                             
%ripplePowerPeak_shift spikeDensityPeak_shift ripplePowerMean_shift spikeDensityMean_shift thetaPowerMean_shift posteriorSpreadMean_shift jumpSizeMean_shift meanSpikes_shift numCells_shift 

%40        41           42            43       44  45  46
%trialType lickingEvent drinkingEvent replayID rat day sessionNum




    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
        % plot(x,y,'k','linewidth',2)
        % xlabel({'num. of tetrodes','detecting ripples'}), ylabel('ripple probability')
        % set(gca,'fontsize',14)
        % axis square, set(gca,'fontsize',12), set(gcf,'color','w'), axis tight
        % vline(numTetsThr,'r-')


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % subplot(121)
        %     histogram(replays_pvalue(:,1),linspace(0,0.2,20),'normalization','probability','edgecolor','none','FaceColor','b'), 
        %     xlabel({'pvalue','(mean posterior spread)'}), ylabel('fraction of replays'), vline(0.05,'r-'), axis square, set(gca,'fontsize',12), 
        % 
        % subplot(122)
        %     histogram(replays_pvalue(:,2),linspace(0,0.2,20),'normalization','probability','edgecolor','none','FaceColor','b')
        %     xlabel({'pvalue','(mean jump size)'}), vline(0.05,'r-'), ylabel('fraction of replays'), axis square, set(gca,'fontsize',12), set(gcf,'color','w')
        % 
        %     sum(replays_pvalue(:,1)<=0.05 & replays_pvalue(:,2)<=0.05)/size(replays_pvalue(:,1),1)

        replays_stats_nanRemoved = replays_stats; replays_stats_nanRemoved(isnan(replays_stats(:,1)),:) = [];
        ind_sessions = unique(replays_stats_nanRemoved(:,end-2:end),'rows');
        data_fractions = nan(size(ind_sessions,1),4,3); 
        data_fractions_tets = nan(size(ind_sessions,1),4,65,2); 
        data_replayStats_frac = nan(size(ind_sessions,1),1,2,2);
        data_replayStats_1 = nan(size(ind_sessions,1),6,2,2);
        data_replayStats_2 = nan(size(ind_sessions,1),26,2,2);
        data_replayStats_3 = nan(size(ind_sessions,1),2,3,2);
        data_powerSpectrum = nan(size(ind_sessions,1),1000,2,2);
        data_powerSpectrum_shift = nan(size(ind_sessions,1),1000,2,2);
        data_powerSpectrum_shuffle = nan(size(ind_sessions,1),1000,2,2);
        data_numTets = nan(size(ind_sessions,1),65);
        data_numTets_shift = nan(size(ind_sessions,1),65);
        data_duration = nan(size(ind_sessions,1),65);
        data_posteriorSpread = nan(size(ind_sessions,1),65);
        data_ripplePeak = nan(size(ind_sessions,1),65);
        data_numTets_shuffle = nan(size(ind_sessions,1),65);
        data_ripplePowerVsSpikeDensity = nan(size(ind_sessions,1),14);
        numReplays = nan(size(ind_sessions,1),4);
        for i = 1:size(ind_sessions,1)
            % ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows'));
            ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows') & replays_stats(:,end-4)==1);
            % ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows') & replays_stats(:,end-4)==1 & replays_pvalue(:,1)<=0.05 & replays_pvalue(:,2)<=0.05);
            replays_stats_sub = replays_stats(ind_session,:);
                replays_stats_sub(:,5:6) = abs(replays_stats_sub(:,5:6)); %take absolute value of the angles
            replays_powerSpectrum_sub = replays_powerSpectrum(ind_session,:);
            replays_powerSpectrum_shuffle_sub = replays_powerSpectrum_shuffle(ind_session,:);
            replays_tets_sub = replays_tets(ind_session,:,:);
            replays_numRippleEvents_tets_shuffle_sub = replays_numRippleEvents_tets_shuffle(ind_session,:,:);
            replays_ripplePowerPeak_tets_shuffle_sub = replays_ripplePowerPeak_tets_shuffle(ind_session,:,:);
            replays_powerSpectrum_shift_sub = replays_powerSpectrum_shift(ind_session,:);
            replays_tets_shift_sub = replays_tets_shift(ind_session,:,:);

            bins = -0.5:1:64.5; 
            A = replays_tets_sub(:,:,2); B = replays_tets_sub(:,:,3); A(B>=pvalueThr) = nan;
            data_numTets(i,:) = histcounts(nansum(A>0,2),bins,'normalization','probability');

            A = replays_tets_shift_sub(:,:,2); B = replays_tets_shift_sub(:,:,3); A(B>pvalueThr) = nan;
            data_numTets_shift(i,:) = histcounts(nansum(A>0,2),bins,'normalization','probability');

            for j = 0:64
                A = replays_tets_sub(:,:,2); B = replays_tets_sub(:,:,3); A(B>=pvalueThr) = nan;
                ind = find(nansum(A>0,2)==j & ~isnan(replays_stats_sub(:,1)));
                data_duration(i,j+1) = nanmean(replays_stats_sub(ind,7));
                data_posteriorSpread(i,j+1) = nanmean(replays_stats_sub(ind,18));
                if ~isempty(ind)
                    data_ripplePeak_sub = nan(length(ind),1);
                    for k = 1:length(ind)
                        data_ripplePeak_sub(k) = nanmean(replays_tets_sub(ind(k),replays_tets_sub(ind(k),:,3)<pvalueThr,1));
                    end
                    data_ripplePeak(i,j+1) = nanmean(data_ripplePeak_sub);
                end
            end

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
                    numReplays(i,:) = [sum(ind_neither) ind_sessions(i,:)];
                end

                % %rat speed matching
                %     edges_ratSpeed = linspace(0,3,12);
                %     index_neither = find(ind_neither); index_either = find(ind_either); 
                %     [index_neither_ratSpeedMatched,index_either_ratSpeedMatched,d1,d2] = compute_matchedHistogram(replays_stats_sub(index_neither,4),replays_stats_sub(index_either,4),edges_ratSpeed);
                %     ind_neither = index_neither(index_neither_ratSpeedMatched);
                %     ind_either = index_either(index_either_ratSpeedMatched);

                % %replay duration matching
                %     edges_replayDuration = linspace(0,2,12);
                %     index_neither = find(ind_neither); index_either = find(ind_either); 
                %     [index_neither_replayDurationMatched,index_either_replayDurationMatched,d1,d2] = compute_matchedHistogram(replays_stats_sub(index_neither,7),replays_stats_sub(index_either,7),edges_replayDuration);
                %     ind_neither = index_neither(index_neither_replayDurationMatched);
                %     ind_either = index_either(index_either_replayDurationMatched);

                data_powerSpectrum(i,:,1,k) = mean(replays_powerSpectrum_sub(ind_neither,:),1,'omitnan');
                data_powerSpectrum(i,:,2,k) = mean(replays_powerSpectrum_sub(ind_either,:),1,'omitnan');
                data_powerSpectrum_shuffle(i,:,1,k) = mean(replays_powerSpectrum_shuffle_sub(ind_neither,:),1,'omitnan');
                data_powerSpectrum_shuffle(i,:,2,k) = mean(replays_powerSpectrum_shuffle_sub(ind_either,:),1,'omitnan');
                data_powerSpectrum_shift(i,:,1,k) = mean(replays_powerSpectrum_shift_sub(ind_neither,:),1,'omitnan');
                data_powerSpectrum_shift(i,:,2,k) = mean(replays_powerSpectrum_shift_sub(ind_either,:),1,'omitnan');

                data_replayStats_frac(i,1,1,k) = sum(ind_neither)/size(replays_stats_sub,1);
                data_replayStats_frac(i,1,2,k) = sum(ind_either)/size(replays_stats_sub,1);
                data_replayStats_1(i,1:end,1,k) = mean(replays_stats_sub(ind_neither,[1 3:4 7 8 6]),1,'omitnan');
                data_replayStats_1(i,1:end,2,k) = mean(replays_stats_sub(ind_either,[1 3:4 7 8 6]),1,'omitnan');
                data_replayStats_2(i,:,1,k) = mean(replays_stats_sub(ind_neither,[14:39]),1,'omitnan');
                data_replayStats_2(i,:,2,k) = mean(replays_stats_sub(ind_either,[14:39]),1,'omitnan');
                data_replayStats_2(i,:,1,k) = mean(replays_stats_sub(ind_neither,[14:39]),1,'omitnan');
                data_replayStats_2(i,:,2,k) = mean(replays_stats_sub(ind_either,[14:39]),1,'omitnan');

                % data_replayStats_3(i,1,1,k) = mean(replays_meanPosteriorSpread_sub(ind_neither,1),1,'omitnan');
                % data_replayStats_3(i,1,2,k) = mean(replays_meanPosteriorSpread_sub(ind_either,1),1,'omitnan');
                % data_replayStats_3(i,1,3,k) = mean(replays_meanPosteriorSpread_sub(:,2),1,'omitnan');
                % 
                % data_replayStats_3(i,2,1,k) = mean(replays_meanJumpSize_sub(ind_neither,1),1,'omitnan');
                % data_replayStats_3(i,2,2,k) = mean(replays_meanJumpSize_sub(ind_either,1),1,'omitnan');
                % data_replayStats_3(i,2,3,k) = mean(replays_meanJumpSize_sub(:,2),1,'omitnan');
            end

            numTetsThr_vec = 1:65;
            for j = 1:length(numTetsThr_vec)
                for k = 1:2
                    if k==1
                        %filter for individual tet ripples only and bursts
                        A = replays_tets_sub(:,:,2); B = replays_tets_sub(:,:,3); A(B>=pvalueThr) = nan;
                        ind_neither = nansum(A>0,2)<numTetsThr_vec(j) & replays_stats_sub(:,10)==0;
                        ind_both = nansum(A>0,2)>=numTetsThr_vec(j) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                        ind_ripp = nansum(A>0,2)>=numTetsThr_vec(j) & replays_stats_sub(:,10)==0;
                        ind_burst = nansum(A>0,2)<numTetsThr_vec(j) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                    else
                        %filter for individual and avg tet ripples and bursts
                        A = replays_tets_sub(:,:,2); B = replays_tets_sub(:,:,3); A(B>=pvalueThr) = nan;
                        ind_neither = (replays_stats_sub(:,9)==0 & (nansum(A>0,2)<numTetsThr_vec(j))) & replays_stats_sub(:,10)==0;
                        ind_both = ((replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr_vec(j)) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                        ind_ripp = ((replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr_vec(j)) & replays_stats_sub(:,10)==0;
                        ind_burst = (replays_stats_sub(:,9)==0 & nansum(A>0,2)<numTetsThr_vec(j)) & (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);
                    end
                    data_fractions_tets(i,:,j,k) = [sum(ind_neither) sum(ind_both) sum(ind_ripp) sum(ind_burst)]/size(replays_stats_sub,1);                    
                end
            end

            %compute mean ripple power as a function of spike density  
                numBins = 15;  
                edges_peakSpikeDensity = linspace(0,20,numBins); centers_peakSpikeDensity = edges_peakSpikeDensity(1:end-1)+mean(diff(edges_peakSpikeDensity))/2;
                for j = 1:length(centers_peakSpikeDensity)
                    ind = find(replays_stats_sub(:,14)>edges_peakSpikeDensity(j) & replays_stats_sub(:,14)<=edges_peakSpikeDensity(j+1));
                    if length(ind)>0
                        data_ripplePowerVsSpikeDensity(i,j) = mean(replays_stats_sub(ind,13),'omitnan');
                    end
                end

        end

        %number of replays w/ vs w/o ripples/bursts and power spectrum
            k = 2; %k=1 for tetrode averaging, k=2 for individual tetrodes

            subplot(231)
            data_sub = {data_replayStats_frac(:,1,1,k),data_replayStats_frac(:,1,2,k)};
            [~,pvalues] = ttest2(data_sub{1},data_sub{2});
            violin(data_sub,'facecolor',[0 0 1; 1 0 0]);
            set(gca,'xtick',1:length(data_sub),'xticklabel',{'w/o ripple and burst','w/ ripple or burst'})
            sigstar([1 2],pvalues)
            xlim([0 length(data_sub)+1]), axis square, ylabel('fraction of replays'), xtickangle(30)
            pbaspect([2 1 1])

            subplot(232)
            bh = bar([[nanmean(data_fractions(:,1,k)) 0 0 0];[0 nanmean(data_fractions(:,2:end,k))]],'stacked');
            ylabel('fraction of replays')
            set(gca,'xtick',1:size(data_fractions,2),'xticklabel',{'w/o ripple and burst','w/ ripple or burst'})
            set(gca,'fontsize',12), axis square, xlim([0.25 2.75])
            legend({'w/o ripple and burst','w/ ripple and burst','ripple only','burst only'})
            xtickangle(30)
            set(gcf,'color','w')

            set(bh, 'FaceColor', 'Flat')
            bh(1).CData = [0 0 1]; bh(1).FaceAlpha = 0.6; 
            bh(2).CData = [1 0 0]; bh(2).FaceAlpha = 0.6;   
            bh(3).CData = [1 0.4 0.4]; bh(3).FaceAlpha = 0.6; 
            bh(4).CData = [1 0.6 0.6]; bh(4).FaceAlpha = 0.6;  

        %power spectrum
            k = 2;
            subplot(233)
            colors = [0 0 1; 1 0 0; 0 1 0; 0.8 0.8 0];
            plot(f(1:1000),nanmean(data_powerSpectrum(:,:,1,k)),'color',[colors(1,:) 0.6],'linewidth',3)
            hold on, plot(f(1:1000),nanmean(data_powerSpectrum(:,:,2,k)),'color',[colors(2,:) 0.6],'linewidth',3), hold off
            hold on, plot(f(1:1000),nanmean([data_powerSpectrum_shift(:,:,1,k); data_powerSpectrum_shift(:,:,2,k)]),'color',[colors(3,:) 0.6],'linestyle',':','linewidth',3), hold off
            hold on, plot(f(1:1000),nanmean([data_powerSpectrum_shuffle(:,:,1,k); data_powerSpectrum_shuffle(:,:,2,k)]),'color',[colors(4,:) 0.6],'linestyle',':','linewidth',3), hold off
            % hold on, plot(f(1:1000),nanmean([data_powerSpectrum_shift(:,:,1,k)]),'color',[colors(3,:) 0.6],'linestyle',':','linewidth',3), hold off
            % hold on, plot(f(1:1000),nanmean([data_powerSpectrum_shuffle(:,:,1,k)]),'color',[colors(4,:) 0.6],'linestyle',':','linewidth',3), hold off
            xlabel('frequency'), ylabel('power'), axis square, set(gca,'fontsize',12), box on, xlim([f(1) f(1000)]), axis tight
            vline([100 220],'r:')
            legend({'w/o ripple and burst','w/ ripple or burst','post-replay','random'})
            set(gcf,'color','w')
            axis square, %pbaspect([3 2 1])

        %event duration, with vs. without ripp
            durations = linspace(0,2,20);
            A = replays_tets(:,:,2); B = replays_tets(:,:,3); A(B>pvalueThr) = nan;
            ind_neither = find(nansum(A>0,2)<numTetsThr & replays_stats(:,9)==0 & replays_stats(:,10)==0 & replays_pvalue(:,1)<=0.05 & replays_pvalue(:,2)<=0.05);
            ind_either = find((nansum(A>0,2)>=numTetsThr | (replays_stats(:,9)>0 & replays_stats(:,11)<pvalueThr) | (replays_stats(:,10)>0 & replays_stats(:,12)<pvalueThr)) & replays_pvalue(:,1)<=0.05 & replays_pvalue(:,2)<=0.05);
            for i = 1:2
                subplot(2,3,i+3)
                histogram(replays_stats(ind_either,7),durations,'edgecolor','none','normalization','probability','facecolor','r')
                hold on, histogram(replays_stats(ind_neither,7),durations,'edgecolor','none','normalization','probability','facecolor','b'), hold off
                axis square, xlabel('duration (s)'), ylabel('fraction of events'), set(gca,'fontsize',12)
                if i==1, legend('w/ ripple or burst','w/o ripple and burst'), end
                xlim([min(durations) max(durations)])
                if i==2, set(gca,'yscale','log'), end
                pbaspect([2 1 1])
            end


        %stats, with vs. without ripp
            k = 2;
            names_2 = {{'spike density peak','(z-scored)'},'posterior spread','jump size','spikes per sec','num cells per sec',{'4-12 Hz power','(z-scored)'}};
            j_vec = [14 18 19 20 21 17]-13;
            for jj = 1:6
                subplot(2,3,jj)
                j = j_vec(jj);

                data_sub = {data_replayStats_2(:,j,1,k),data_replayStats_2(:,j,2,k),[data_replayStats_2(:,j+18,1,k); data_replayStats_2(:,j+18,2,k)],[data_replayStats_2(:,j+9,1,k); data_replayStats_2(:,j+9,2,k)]};
                
                if jj == 3, data_sub{4} = data_sub{3}; end

                pvalues = nan(3,1);
                [~,pvalues(1)] = ttest2(data_sub{1},data_sub{2});
                [~,pvalues(2)] = ttest2(data_sub{1},data_sub{3});
                [~,pvalues(3)] = ttest2(data_sub{1},data_sub{4});
                violin(data_sub,'facecolor',colors);
                sigstar({[1 2],[1 3],[1 4]},pvalues)
                set(gca,'xtick',1:length(data_sub),'xticklabel',{'w/o ripple and burst','w/ ripple or burst','post-replay','random'})

                xlim([0 length(data_sub)+1]), ylabel(names_2{jj}), xtickangle(30), pbaspect([2 1 1])
                if jj==1, hline(3,'k--'), end

                ax = gca; % Get current axes
                ax.TickLength = [0.02, 0.02];
            end
            set(findobj(gcf,'type','axes'),'FontSize',12), set(gcf,'color','w')


            names_1 = {{'time since start','of session (min)'},{'time before','movement (sec)'},'rat speed (cm/sec)','duration (sec)','replay speed (cm/sec)',{'abs. displacement','rel. to future behavior (deg)'}};
            for j = 1:6
                subplot(2,3,j)
                data_sub = {data_replayStats_1(:,j,1,k),data_replayStats_1(:,j,2,k)};
                if j==1, data_sub = {data_replayStats_1(:,j,1,k)/60,data_replayStats_1(:,j,2,k)/60}; end
                if j==3, binSize = 2; data_sub = {binSize*data_replayStats_1(:,j,1,k),binSize*data_replayStats_1(:,j,2,k)}; end
                if j==6, data_sub = {180/pi*data_replayStats_1(:,j,1,k),180/pi*data_replayStats_1(:,j,2,k)}; end
                [~,pvalues] = ttest2(data_sub{1},data_sub{2});
                violin(data_sub,'facecolor',[0 0 1; 1 0 0]);
                set(gca,'xtick',1:length(data_sub),'xticklabel',{'w/o ripple and burst','w/ ripple or burst'})
                sigstar([1 2],pvalues)
                xlim([0 length(data_sub)+1]), axis square, ylabel(names_1{j}), xtickangle(30), pbaspect([2 1 1])
                ax = gca; % Get current axis
                ax.FontSize = 14; % Set tick label font size larger
                ax.TickLength = [0.02, 0.02];
            end
            set(findobj(gcf,'type','axes'),'FontSize',12), set(gcf,'color','w')

          
        %number of replays and replay duration as a function of number of tetrodes detecting a ripple
            set(gcf,'color','w')
            ind = find(nansum(data_numTets>0)>0);
            % ind = find(nansum(data_numTets>0)>15); 
            vec = 0:64;
            subplot(231)
            shadedErrorBar(vec(ind),nanmean(data_numTets(:,ind)),nanstd(data_numTets(:,ind))./sqrt(size(ind_sessions,1)),'lineprops',{'k','linewidth',2})
            hold on, shadedErrorBar(vec(ind),nanmean(data_numTets_shift(:,ind)),nanstd(data_numTets_shift(:,ind))./sqrt(size(ind_sessions,1)),'lineprops',{'color',colors(3,:),'linewidth',2}), hold off
            xlabel({'num. of tetrodes','detecting ripples'}), ylabel('fraction of replays'), axis square, set(gca,'fontsize',14), box on
            xlim([-1 65]), ylim([0 0.09])

            pvalues = nan(1,65);
            for i = 1:length(ind)
                [~,pvalues(ind(i))] = ttest(data_numTets(:,ind(i)),data_numTets_shift(:,ind(i)));
            end
            pvalues_binary = nan(size(pvalues)); pvalues_binary(pvalues<0.05) = 1;
            hold on, plot(0:64,0.08*pvalues_binary,'k*'), hold off
            legend('replay','post replay')

            subplot(232)
            shadedErrorBar(shiftPeriod_vec,nanmean(numRippleEvents_shift_vec,1),nanstd(numRippleEvents_shift_vec,1)./sqrt(size(numRippleEvents_shift_vec,1)),'lineprops',{'k','linewidth',2})
            axis square, set(gca,'fontsize',12), xlabel('post-replay time (sec)'), ylabel('probability of ripple')
            xlim([0 2.5]), box on, set(gcf,'color','w'), 
            vline(0.75,'g--')

            subplot(233)
            shadedErrorBar(0:64,nanmean(data_ripplePeak),nanstd(data_ripplePeak)./sqrt(size(ind_sessions,1)),'lineprops',{'k','linewidth',2})
            xlabel({'num. of tetrodes','detecting ripples'}), ylabel('ripple peak'), axis square, set(gca,'fontsize',12), box on
            xlim([-1 65]), 

            subplot(234)
            shadedErrorBar(0:64,nanmean(data_duration),nanstd(data_duration)./sqrt(size(ind_sessions,1)),'lineprops',{'k','linewidth',2})
            xlabel({'num. of tetrodes','detecting ripples'}), ylabel('replay duration'), axis square, set(gca,'fontsize',12), box on
            xlim([-1 65]), ylim([0 0.6])

            subplot(235)
            shadedErrorBar(0:64,nanmean(data_posteriorSpread),nanstd(data_posteriorSpread)./sqrt(size(ind_sessions,1)),'lineprops',{'k','linewidth',2})
            xlabel({'num. of tetrodes','detecting ripples'}), ylabel('replay posterior spread'), axis square, set(gca,'fontsize',12), box on
            xlim([-1 65]), ylim([0 3])

            subplot(236)
            x = 0:20;
            y = binopdf(x,numDetectors,probSucess);
            plot(x,y,'k','linewidth',2)
            xlabel({'num. of tetrodes','detecting ripples'}), ylabel('ripple probability')
            set(gca,'fontsize',14)
            numTetsThr = InterX([x; cumsum(y)/sum(y)]',[min(x) max(x); 0.95 0.95]'); numTetsThr = round(numTetsThr(1));
            axis square, set(gca,'fontsize',12), set(gcf,'color','w'), axis tight
            vline(numTetsThr,'r--')


        %ripple peak vs spike density peak
            numBins = 40;
            edges_peakSpikeDensity = linspace(0,20,numBins); centers_peakSpikeDensity = edges_peakSpikeDensity(1:end-1)+mean(diff(edges_peakSpikeDensity))/2;
            data_ripplePowerVsSpikeDensity = nan(2,length(centers_peakSpikeDensity));
            for j = 1:length(centers_peakSpikeDensity)
                ind = find(replays_stats(:,14)>edges_peakSpikeDensity(j) & replays_stats(:,14)<=edges_peakSpikeDensity(j+1));
                data_ripplePowerVsSpikeDensity(1,j) = mean(replays_stats(ind,13),'omitnan');
                data_ripplePowerVsSpikeDensity(2,j) = length(ind);
            end
            scatter(replays_stats(:,14),replays_stats(:,13),'k.')
            ind = find(data_ripplePowerVsSpikeDensity(2,:)>60);
            hold on, plot(centers_peakSpikeDensity(ind),data_ripplePowerVsSpikeDensity(1,ind),'r','linewidth',4), hold off
            axis([-1 9 -1 20])
            axis square, set(gca,'fontsize',14), set(gcf,'color','w'), box on
            xlabel('spike density peak (z-scored)'), ylabel('ripple power peak (z-scored)')

      
        %plot ripple phase shift as a function of tetrode distance
            load ripplePhaseOffset_vs_tetrodePosition
            edges = linspace(0,2,10); centers = [edges(1:end-1) + mean(diff(edges))/2 3];
            data = nan(length(data_tetrodePositions),length(edges));
            for i = 1:length(data_tetrodePositions)
                for j = 1:length(edges)
                    if j<length(edges)
                        ind = find(data_tetrodePositions{i}(:,1)>edges(j) & data_tetrodePositions{i}(:,1)<=edges(j+1));
                    else 
                        ind = find(isnan(data_tetrodePositions{i}(:,1)));
                    end
                    if isempty(ind), continue, end
                    data(i,j) = nanmean(nanmean(abs(data_tetrodePositions{i}(ind,2:end-3))));
                end
            end
            plot(centers,nanmean(data),'k:')
            shadedErrorBar(centers(1:end-1),nanmean(data(:,1:end-1)),nanstd(data(:,1:end-1))./sqrt(length(data_tetrodePositions)),'lineprops',{'linewidth',2,'color','k'})
            hold on, errorbar(centers(end),nanmean(data(:,end)),nanstd(data(:,end))./sqrt(length(data_tetrodePositions)),'k.','linewidth',1,'markersize',10), hold off
            xlim([0 3.2])
            set(gca,'xtick',[0:1:2,centers(end)],'xticklabel',{0:1:2,'other hem.'}), xtickangle(0)
            axis square, set(gca,'fontsize',14), set(gcf,'color','w'), box on
            xlabel('distance between tetrodes (mm)'), ylabel('abs. phase shift (rad)')

            for rat = 1:3
                if rat==1
                    ind = 1:12;
                elseif rat==2
                    ind = 13:29;
                else
                    ind = 30:37;
                end
                subplot(1,3,rat)
                plot(centers,nanmean(data(ind,:)),'k:')
                shadedErrorBar(centers(1:end-1),nanmean(data(ind,1:end-1)),nanstd(data(ind,1:end-1))./sqrt(size(data(ind,end),1)),'lineprops',{'linewidth',2,'color','k'})
                hold on, errorbar(centers(end),nanmean(data(ind,end)),nanstd(data(ind,end))./sqrt(size(data(ind,end),1)),'k.','linewidth',1,'markersize',10), hold off
                xlim([0 3.2])
                set(gca,'xtick',[0:1:2,centers(end)],'xticklabel',{0:1:2,'other hem.'}), xtickangle(0)
                axis square, set(gca,'fontsize',24), set(gcf,'color','w'), box on
                xlabel({'distance between','tetrodes (mm)'}), ylabel('abs. phase shift (rad)')
                title(strcat('rat',num2str(rat)))
            end
        

    
         %plot replay distance to rat
            pvalueThr = 0.05;
            Rat_Names = {'Billy3','Curly2','Goethe2'};
            ind_sessions = unique(replays_stats_nanRemoved(:,end-2:end),'rows');
            distanceToRat = [];
            for i = 1:size(ind_sessions,1)
                rat = ind_sessions(i,1);
                day = ind_sessions(i,2);
                sessionNum = ind_sessions(i,3); sessionNum_decoder = sessionNum;
    
                cd(strcat(Rat_Names{rat},'/processed'))
                if rat == 1 %Billy3
                    dayFiles = {'20181205','20181206','20181207','20181208','20181212'};%,'20181218'};
                elseif rat == 2 %Curly2
                    dayFiles = {'20191007','20191008','20191009','20191010','20191011','20191013'};
                elseif rat == 3 %Goethe2
                    dayFiles = {'20180607','20180608','20180609','20180614'};
                end
                cd(dayFiles{day})
                load Experiment_Information.mat
                load Analysis_Information.mat
                load replayEvents
                cd ../../..
    
                ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows'));
                replays_stats_sub = replays_stats(ind_session,:);
                replays_tets_sub = replays_tets(ind_session,:,:);
    
                replay_dispersionThr = 12; 
                load_ind_replayEvents
        
                A = replays_tets_sub(:,:,2); B = replays_tets_sub(:,:,3); A(B>=pvalueThr) = nan;
                ind_neither = replays_stats_sub(:,9)==0 & nansum(A>0,2)<numTetsThr & replays_stats_sub(:,10)==0;
                ind_either = (replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr | (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);

                ind_neither = find(ind_neither);
                ind_either = find(ind_either);
                ind_all = replays_stats_sub(:,40);
    
                distanceToRat_sub = nan(length(ind_all),2);
                for j = 1:length(ind_all)
                    replay_ratLoc = ratLoc(ind_all(j),:);
                    [~,replay_NaN] = compute_replaysConcatenation(decoder_replay(sessionNum),ind_all(j));
                    replay_startLoc = replay_NaN(1,2:3);
                    replay_endLoc = replay_NaN(end,2:3);

                    distanceToRat_sub(j,1) = sqrt((replay_ratLoc(1)-replay_startLoc(1)).^2 + (replay_ratLoc(2)-replay_startLoc(2)).^2);
                    distanceToRat_sub(j,2) = sqrt((replay_ratLoc(1)-replay_endLoc(1)).^2 + (replay_ratLoc(2)-replay_endLoc(2)).^2);
                end

                distanceToRat = [distanceToRat; nanmean(distanceToRat_sub) nanmean(distanceToRat_sub(ind_neither,:)) nanmean(distanceToRat_sub(ind_either,:))];
            end

            violin(binSize*distanceToRat(:,1:2),'facecolor',[0 0 0; 0 0 0]);
            set(gca,'xtick',1:2,'xticklabel',{'replay start to rat','replay end to rat'})
            [~,pvalue] = ttest2(distanceToRat(:,1),distanceToRat(:,2));
            sigstar({[1 2]},pvalue)
            axis square, set(gcf,'color','w'), ylabel('distance (cm)'), set(gca,'fontsize',14)
            

         %plot replays, w/ vs w/o ripples
            load replayEvents_SWRfield_6
            pvalueThr = 0.05;
            Rat_Names = {'Billy3','Curly2','Goethe2'};
            replays_stats_nanRemoved = replays_stats; replays_stats_nanRemoved(isnan(replays_stats(:,1)),:) = [];
            ind_sessions = unique(replays_stats_nanRemoved(:,end-2:end),'rows');
            for i = 4:size(ind_sessions,1)
                rat = ind_sessions(i,1);
                day = ind_sessions(i,2);
                sessionNum = ind_sessions(i,3); sessionNum_decoder = sessionNum;
    
                cd(strcat(Rat_Names{rat},'/processed'))
                if rat == 1 %Billy3
                    dayFiles = {'20181205','20181206','20181207','20181208','20181212'};%,'20181218'};
                elseif rat == 2 %Curly2
                    dayFiles = {'20191007','20191008','20191009','20191010','20191011','20191013'};
                elseif rat == 3 %Goethe2
                    dayFiles = {'20180607','20180608','20180609','20180614'};
                end
                cd(dayFiles{day})
                load Experiment_Information.mat
                load Analysis_Information.mat
                load replayEvents
                cd ../../..
    
                ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows'));
                replays_stats_sub = replays_stats(ind_session,:);
                replays_tets_sub = replays_tets(ind_session,:,:);
    
                replay_dispersionThr = 12; 
                load_ind_replayEvents
        
                A = replays_tets_sub(:,:,2); B = replays_tets_sub(:,:,3); A(B>=pvalueThr) = nan;
                ind_neither = replays_stats_sub(:,9)==0 & nansum(A>0,2)<numTetsThr & replays_stats_sub(:,10)==0;
                ind_either = (replays_stats_sub(:,9)>0 & replays_stats_sub(:,11)<pvalueThr) | nansum(A>0,2)>=numTetsThr | (replays_stats_sub(:,10)>0 & replays_stats_sub(:,12)<pvalueThr);

                ind_neither = find(ind_neither);
                ind_either = find(ind_either);

                ind_replay_neither = replays_stats_sub(ind_neither,40);
                ind_replay_either = replays_stats_sub(ind_either,40);
                ind_all = replays_stats_sub(:,40);
    
                [~,replays_NaN_neither] = compute_replaysConcatenation(decoder_replay(sessionNum),ind_replay_neither);
                [~,replays_NaN_either] = compute_replaysConcatenation(decoder_replay(sessionNum),ind_replay_either);
                [~,replays_NaN_all] = compute_replaysConcatenation(decoder_replay(sessionNum),ind_all);

                rippleField = Data(i).LFP.replays_amp_mat;

                subplot(121)
                imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3)), colormap(gray)
                hold on, imagesc((Data(i).LFP.replays_amp_mat),'alphadata',~isnan(Data(i).LFP.replays_amp_mat)), hold off
                axis square, colorStyle = 'k'; plot_mazeProperties, caxis([0 2])
                hold on, plot(replays_NaN_either(:,2),replays_NaN_either(:,3),'r','linewidth',2), hold off
                hold on, plot(replays_NaN_neither(:,2),replays_NaN_neither(:,3),'b','linewidth',2), hold off
                plot_mazeProperties

                for j = 1:length(ind_neither)
                    [~,replay_NaN_neither,replay_NaNremoved_neither] = compute_replaysConcatenation(decoder_replay(sessionNum),ind_replay_neither(j));

                    subplot(122)
                    plot(replay_NaNremoved_neither(:,2),replay_NaNremoved_neither(:,3),'linewidth',4)
                    plot_mazeProperties
                    keyboard
                end
                

                keyboard
            end


        %fraction of ripple/burst-less replays as a function of tetrode threshold
            k = 2;
            shadedErrorBar(0:64,squeeze(nanmean(data_fractions_tets(:,1,:,k),1)),squeeze(nanstd(data_fractions_tets(:,1,:,k),1)/length(ind_sessions)),'lineprops',{'color','k','linewidth',2}) %[0 0.4470 0.7410]
            axis square, set(gca,'fontsize',14), xlabel('threshold on number of tetrodes'), ylabel({'fraction of replays','w/o ripple and burst'}), box on
            xlim([-1 65]), ylim([0 0.4])
            % vline(numTetsThr,'g--')


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %replay duration vs num ripple events
        % cplot(replays_stats(:,12),replays_stats(:,8)+0.1*(2*rand(size(replays_stats(:,8)))-1),replays_stats(:,23),'o','linewidth',2), 
        plot(replays_stats(:,12)+0.02*(2*rand(size(replays_stats(:,8)))-1),replays_stats(:,8)+0.3*(2*rand(size(replays_stats(:,8)))-1),'.','markersize',8), 
        xlabel('duration'), ylabel('# ripples'), xlim([0 1]), %axis square, 
        durations = linspace(0,1,10);
        X = [ones(size(replays_stats(:,12),1),1) replays_stats(:,12)];
        b = X\replays_stats(:,8); y_fit = X*b;
        hold on, plot(durations,b(1)+b(2)*durations,'r','LineWidth',2), hold off
        [r,p] = compute_corrCoef(replays_stats(:,[8 12]));
        yl = ylim; xl = xlim; text(xl(1)+diff(xl)*0.05,yl(2)-diff(yl)*0.05,strcat('r=',num2str(compute_round(r,100)),load_pstar(p)),'fontsize',12,'VerticalAlignment','top','HorizontalAlignment','left')
        % colorbar, caxis([0 0.5]), colormap(cool), ylabel(colorbar,'p value','rotation',270,'fontsize',14)



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    data = cell(1,3);
    data_ripplePower = cell(1,4);
    data_spikeDensity = cell(1,4);
    data_ripplePower_elapsedTime = cell(1,2);
    data_spikeDensity_elapsedTime = cell(1,2);
    data_ratSpeedVsTimeBeforeMovement = [];
    data_replaySpeedVsRipplePower = [];
    data_replaySpeedVsSpikeDensity = [];
    data_ripplePowerVsSpikeDensity = [];
    data_replaySpeedVsPosteriorSpread= [];
    ind_sessions = unique(replays_stats(:,end-2:end),'rows');
    for i = 1:length(ind_sessions)
        ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows') & replays_stats(:,17)==1);
        replays_stats_sub = replays_stats(ind_session,:);
        ind_woRipples = find(replays_stats_sub(:,8)==0 & replays_stats_sub(:,23)>0.05);
        ind_wRipples = find(replays_stats_sub(:,8)>0 & replays_stats_sub(:,23)<0.05);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data{1} = [data{1}; mean(replays_stats_sub(ind_woRipples,[13:14 3 12 15:16 27:28 1]),1,'omitnan') length(ind_woRipples)];
            data{2} = [data{2}; mean(replays_stats_sub(ind_wRipples,[13:14 3 12 15:16 27:28 1]),1,'omitnan') length(ind_wRipples)];
            data{3} = [data{3}; mean(replays_stats_sub(ind_wRipples,20:21),1,'omitnan') nan(1,8)];

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            edges_timeBeforeMovement = linspace(0,10,14); centers_timeBeforeMovement = edges_timeBeforeMovement(1:end-1) + mean(diff(edges_timeBeforeMovement))/2;
            data_ratSpeedVsTimeBeforeMovement_sub = nan(2,length(edges_timeBeforeMovement)-1);
            for j = 1:length(edges_timeBeforeMovement)-1
                ind = find(replays_stats_sub(:,27)>edges_timeBeforeMovement(j) & replays_stats_sub(:,27)<edges_timeBeforeMovement(j+1));
                data_ratSpeedVsTimeBeforeMovement_sub(j) = mean(replays_stats_sub(ind,15),'omitnan');
            end
            data_ratSpeedVsTimeBeforeMovement = [data_ratSpeedVsTimeBeforeMovement; data_ratSpeedVsTimeBeforeMovement_sub];

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % numBins = 15;    
            % edges_ripplePower = linspace(0,10,numBins); centers_ripplePower = edges_ripplePower(1:end-1) + mean(diff(edges_ripplePower))/2;
            % edges_posteriorSpread = linspace(0,10,numBins); centers_posteriorSpread = edges_posteriorSpread(1:end-1) + mean(diff(edges_posteriorSpread))/2;
            % 
            % data_replaySpeedVsRipplePower_sub = nan(1,length(edges_ripplePower)-1);
            % data_replaySpeedVsSpikeDensity_sub = nan(1,length(edges_ripplePower)-1);
            % data_ripplePowerVsSpikeDensity_sub = nan(1,length(edges_ripplePower)-1);
            % data_replaySpeedVsPosteriorSpread_sub = nan(1,length(edges_posteriorSpread)-1);
            % for j = 1:numBins-1
            %     ind = find(replays_stats_sub(:,2)>edges_ripplePower(j) & replays_stats_sub(:,2)<edges_ripplePower(j+1));
            %     data_replaySpeedVsRipplePower_sub(j) = mean(replays_stats_sub(ind,13),'omitnan');
            % 
            %     ind = find(replays_stats_sub(:,3)>edges_ripplePower(j) & replays_stats_sub(:,3)<edges_ripplePower(j+1));
            %     data_replaySpeedVsSpikeDensity_sub(j) = mean(replays_stats_sub(ind,13),'omitnan');
            % 
            %     ind = find(replays_stats_sub(:,3)>edges_ripplePower(j) & replays_stats_sub(:,3)<edges_ripplePower(j+1));
            %     data_ripplePowerVsSpikeDensity_sub(j) = mean(replays_stats_sub(ind,4),'omitnan');
            % 
            %     ind = find(replays_stats_sub(:,14)>edges_posteriorSpread(j) & replays_stats_sub(:,14)<edges_posteriorSpread(j+1));
            %     data_replaySpeedVsPosteriorSpread_sub(j) = mean(replays_stats_sub(ind,13),'omitnan');
            % end
            % data_replaySpeedVsSpikeDensity = [data_replaySpeedVsSpikeDensity; data_replaySpeedVsSpikeDensity_sub];
            % data_replaySpeedVsRipplePower = [data_replaySpeedVsRipplePower; data_replaySpeedVsRipplePower_sub];
            % data_ripplePowerVsSpikeDensity = [data_ripplePowerVsSpikeDensity; data_ripplePowerVsSpikeDensity_sub];
            % data_replaySpeedVsPosteriorSpread = [data_replaySpeedVsPosteriorSpread; data_replaySpeedVsPosteriorSpread_sub];

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %compute mean ripple power as a function of spike density  
            numBins = 15;  
            edges_meanSpikeDensity = linspace(0,5,numBins); centers_meanSpikeDensity = edges_meanSpikeDensity(1:end-1)+mean(diff(edges_meanSpikeDensity))/2;
            edges_peakSpikeDensity = linspace(0,20,numBins); centers_peakSpikeDensity = edges_peakSpikeDensity(1:end-1)+mean(diff(edges_peakSpikeDensity))/2;
            data_ripplePowerVsSpikeDensity_sub = nan(1,length(edges_meanSpikeDensity)-1);
            for j = 1:length(edges_meanSpikeDensity)-1
                ind = find(replays_stats_sub(:,5)>edges_peakSpikeDensity(j) & replays_stats_sub(:,5)<edges_peakSpikeDensity(j+1));
                if length(ind)>10
                    data_ripplePowerVsSpikeDensity_sub(j) = mean(replays_stats_sub(ind,4),'omitnan');
                end
            end
            data_ripplePowerVsSpikeDensity = [data_ripplePowerVsSpikeDensity; data_ripplePowerVsSpikeDensity_sub];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % %compute mean ripple power mean/peak as a function of elapsed time
        %     edges_elapsedTime = linspace(0,60*60,12); centers_elapsedTime = edges_elapsedTime(1:end-1)+mean(diff(edges_elapsedTime))/2;
        %     data_ripplePower_sub = nan(2,length(edges_elapsedTime)-1);
        %     data_spikeDensity_sub = nan(2,length(edges_elapsedTime)-1);
        %     for j = 1:length(edges_elapsedTime)-1
        %         ind = find(replays_stats_sub(:,1)>edges_elapsedTime(j) & replays_stats_sub(:,1)<edges_elapsedTime(j+1));
        %         if length(ind)>2
        %             data_ripplePower_sub(1,j) = mean(replays_stats_sub(ind,2),'omitnan');
        %             data_ripplePower_sub(2,j) = mean(replays_stats_sub(ind,4),'omitnan');
        %             data_spikeDensity_sub(1,j) = mean(replays_stats_sub(ind,3),'omitnan');
        %             data_spikeDensity_sub(2,j) = mean(replays_stats_sub(ind,5),'omitnan');
        %         end
        %     end
        %     data_ripplePower_elapsedTime{1} = [data_ripplePower_elapsedTime{1}; data_ripplePower_sub(1,:)];
        %     data_ripplePower_elapsedTime{2} = [data_ripplePower_elapsedTime{2}; data_ripplePower_sub(2,:)];
        %     data_spikeDensity_elapsedTime{1} = [data_spikeDensity_elapsedTime{1}; data_spikeDensity_sub(1,:)];
        %     data_spikeDensity_elapsedTime{2} = [data_spikeDensity_elapsedTime{2}; data_spikeDensity_sub(2,:)];

%         %compute difference in ripple power mean/peak for pairs of events as a function ofs. time difference
%             edges_timesDiff = linspace(0,40,20); centers_timesDiff = edges_timesDiff(1:end-1)+mean(diff(edges_timesDiff))/2;
%             timesDiff_full = pdist(replays_stats_sub(:,1)); timesDiff_full = timesDiff_full(:);
%             ripplePowerMeanDiff_full = pdist(replays_stats_sub(:,2)); ripplePowerMeanDiff_full = ripplePowerMeanDiff_full(:);
%             ripplePowerPeakDiff_full = pdist(replays_stats_sub(:,4)); ripplePowerPeakDiff_full = ripplePowerPeakDiff_full(:);
%             spikeDensityMeanDiff_full = pdist(replays_stats_sub(:,3)); spikeDensityMeanDiff_full = spikeDensityMeanDiff_full(:);
%             spikeDensityPeakDiff_full = pdist(replays_stats_sub(:,5)); spikeDensityPeakDiff_full = spikeDensityPeakDiff_full(:);
%             
%             data_ripplePower_sub = nan(2,length(edges_timesDiff)-1);
%             data_spikeDensity_sub = nan(2,length(edges_timesDiff)-1);
%             for j = 1:length(edges_timesDiff)-1
%                 ind = find(timesDiff_full>edges_timesDiff(j) & timesDiff_full<edges_timesDiff(j+1));
%                 data_ripplePower_sub(1,j) = mean(ripplePowerMeanDiff_full(ind),'omitnan');
%                 data_ripplePower_sub(2,j) = mean(ripplePowerPeakDiff_full(ind),'omitnan');
%                 data_spikeDensity_sub(1,j) = mean(spikeDensityMeanDiff_full(ind),'omitnan');
%                 data_spikeDensity_sub(2,j) = mean(spikeDensityPeakDiff_full(ind),'omitnan');
%             end
%             data_ripplePower{1} = [data_ripplePower{1}; data_ripplePower_sub(1,:)];
%             data_ripplePower{2} = [data_ripplePower{2}; data_ripplePower_sub(2,:)];
%             data_spikeDensity{1} = [data_spikeDensity{1}; data_spikeDensity_sub(1,:)];
%             data_spikeDensity{2} = [data_spikeDensity{2}; data_spikeDensity_sub(2,:)];
% 
%             %shuffles
%                 numShuffles = 100;
%                 data_ripplePower_sub = cell(1,2);
%                 data_spikeDensity_sub = cell(1,2);
%                 for k = 1:numShuffles
%                     randShift = randperm(size(ripplePowerMeanDiff_full,1)); randShift(1);
%                     ripplePowerMeanDiff_full_shuffle = circshift(ripplePowerMeanDiff_full,randShift);
%                     ripplePowerPeakDiff_full_shuffle = circshift(ripplePowerPeakDiff_full,randShift);
%                     spikeDensityMeanDiff_full_shuffle = circshift(spikeDensityMeanDiff_full,randShift);
%                     spikeDensityPeakDiff_full_shuffle = circshift(spikeDensityPeakDiff_full,randShift);
%                 
%                     data_ripplePower_subsub = nan(2,length(edges_timesDiff)-1);
%                     data_spikeDensity_subsub = nan(2,length(edges_timesDiff)-1);
%                     for j = 1:length(edges_timesDiff)-1
%                         ind = find(timesDiff_full>edges_timesDiff(j) & timesDiff_full<edges_timesDiff(j+1));
%                         data_ripplePower_subsub(1,j) = mean(ripplePowerMeanDiff_full_shuffle(ind),'omitnan');
%                         data_ripplePower_subsub(2,j) = mean(ripplePowerPeakDiff_full_shuffle(ind),'omitnan');
%                         data_spikeDensity_subsub(1,j) = mean(spikeDensityMeanDiff_full_shuffle(ind),'omitnan');
%                         data_spikeDensity_subsub(2,j) = mean(spikeDensityPeakDiff_full_shuffle(ind),'omitnan');
%                     end
%                     data_ripplePower_sub{1} = [data_ripplePower_sub{1}; data_ripplePower_subsub(1,:)];
%                     data_ripplePower_sub{2} = [data_ripplePower_sub{2}; data_ripplePower_subsub(2,:)];
%                     data_spikeDensity_sub{1} = [data_spikeDensity_sub{1}; data_spikeDensity_subsub(1,:)];
%                     data_spikeDensity_sub{2} = [data_spikeDensity_sub{2}; data_spikeDensity_subsub(2,:)];
%                 end
%                 data_ripplePower{3} = [data_ripplePower{3}; mean(data_ripplePower_sub{1},'omitnan')];
%                 data_ripplePower{4} = [data_ripplePower{4}; mean(data_ripplePower_sub{2},'omitnan')];
%                 data_spikeDensity{3} = [data_spikeDensity{3}; mean(data_spikeDensity_sub{1},'omitnan')];
%                 data_spikeDensity{4} = [data_spikeDensity{4}; mean(data_spikeDensity_sub{2},'omitnan')];

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %plot replay properties, w/ and w/o ripples

%             ind_posteriorSpread = find(replays_stats_sub(:,25)<0.05);
% 
%             subplot(221)
% %             plot(replays_stats_sub(:,12),replays_stats_sub(:,8)+0.3*(2*rand(size(replays_stats_sub(:,8)))-1),'ko','linewidth',1), 
%             cplot(replays_stats_sub(ind_posteriorSpread,12),replays_stats_sub(ind_posteriorSpread,8)+0.1*(2*rand(size(replays_stats_sub(ind_posteriorSpread,8)))-1),replays_stats_sub(ind_posteriorSpread,23),'o','linewidth',2), 
%             xlabel('duration'), ylabel('# ripples'), axis square, xlim([0 1])
%             durations = linspace(0,1,10);
%             X = [ones(size(replays_stats_sub(ind_posteriorSpread,12),1),1) replays_stats_sub(ind_posteriorSpread,12)];
%             b = X\replays_stats_sub(ind_posteriorSpread,8); y_fit = X*b;
%             hold on, plot(durations,b(1)+b(2)*durations,'r','LineWidth',2), hold off
%             [r,p] = compute_corrCoef(replays_stats_sub(ind_posteriorSpread,[8 12]));
%             yl = ylim; xl = xlim; text(xl(1)+diff(xl)*0.05,yl(2)-diff(yl)*0.05,strcat('r=',num2str(compute_round(r,100)),load_pstar(p)),'fontsize',12,'VerticalAlignment','top','HorizontalAlignment','left')
%             colorbar, caxis([0 0.05]), colormap(cool), ylabel(colorbar,'p value','rotation',270,'fontsize',14)
% 
%             numEvents = nan(1,max(replays_stats_sub(ind_posteriorSpread,8)+1));
%             duration = nan(2,max(replays_stats_sub(ind_posteriorSpread,8)+1));
%             for j = 0:max(replays_stats_sub(ind_posteriorSpread,8))
%                 ind = find(replays_stats_sub(ind_posteriorSpread,8)==j);
%                 numEvents(j+1) = length(ind);
%                 duration(1,j+1) = mean(replays_stats_sub(ind,12),'omitnan');
%                 duration(2,j+1) = std(replays_stats_sub(ind,12),'omitnan');
%             end
%             subplot(222)
%             barh(0:max(replays_stats_sub(ind_posteriorSpread,8)),numEvents,'k')
%             ylabel('# ripples'), xlabel('count'), axis square, ylim([0-1 max(replays_stats_sub(:,8))+1])
% 
%             subplot(223)
%             errorbar(duration(1,:),duration(2,:),'k','LineWidth',2)
%             set(gca,'xticklabel',0:length(numEvents)), xlabel('# ripples'), ylabel('duration (s)'), axis square, xlim([0 length(numEvents)+1]), set(gca,'xtick',1:length(numEvents))
% 
%             subplot(224)
%             bar(1:2,[numEvents(1) sum(numEvents(2:end))],'k')
%             ylabel('count'), axis square, set(gca,'xtick',1:2,'xticklabel',{'w/o ripp.','w/ ripp'}), xlabel(''), xlim([0 3])
% 
%         set(findobj(gcf,'type','axes'),'FontSize',14), set(gcf,'color','w')
%         keyboard
    end



    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %plot ripple power mean/peak as a spike density mean/peak


        data_ripplePowerVsSpikeDensity = [];
        ind_sessions = unique(replays_stats(:,end-2:end),'rows');
        for i = 1:length(ind_sessions)
            ind_session = find(ismember(replays_stats(:,end-2:end),ind_sessions(i,:),'rows') & replays_stats(:,17)==1);
            replays_stats_sub = replays_stats(ind_session,:);
    
            numBins = 15;  
            edges_peakSpikeDensity = linspace(3,10,numBins); centers_peakSpikeDensity = edges_peakSpikeDensity(1:end-1)+mean(diff(edges_peakSpikeDensity))/2;
            data_ripplePowerVsSpikeDensity_sub = nan(1,length(edges_meanSpikeDensity)-1);
            for j = 1:length(edges_meanSpikeDensity)-1
                ind = find(replays_stats_sub(:,5)>edges_peakSpikeDensity(j) & replays_stats_sub(:,5)<edges_peakSpikeDensity(j+1));
                if length(ind)>0
                    data_ripplePowerVsSpikeDensity_sub(j) = mean(replays_stats_sub(ind,4),'omitnan');
                end
            end
            data_ripplePowerVsSpikeDensity = [data_ripplePowerVsSpikeDensity; data_ripplePowerVsSpikeDensity_sub];
        end


        shadedErrorBar(centers_peakSpikeDensity,mean(data_ripplePowerVsSpikeDensity,'omitnan'),std(data_ripplePowerVsSpikeDensity,'omitnan')./sqrt(size(data_ripplePowerVsSpikeDensity,1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        xlabel('peak spike density'), ylabel('peak ripple power'), axis square
        set(findobj(gcf,'type','axes'),'FontSize',12), set(gcf,'color','w')



        numBins = 8;  
        edges_peakSpikeDensity = linspace(1,10,numBins); centers_peakSpikeDensity = edges_peakSpikeDensity(1:end-1)+mean(diff(edges_peakSpikeDensity))/2;
        data_ripplePowerVsSpikeDensity = nan(3,length(edges_peakSpikeDensity)-1);
        for j = 1:length(edges_peakSpikeDensity)-1
            ind = find(replays_stats(:,5)>edges_peakSpikeDensity(j) & replays_stats(:,5)<edges_peakSpikeDensity(j+1));
            if length(ind)>0
                data_ripplePowerVsSpikeDensity(1,j) = nanmean(replays_stats(ind,4));
                data_ripplePowerVsSpikeDensity(2,j) = nanstd(replays_stats(ind,4))./sqrt(length(ind));
                data_ripplePowerVsSpikeDensity(3,j) = length(ind);
            end
        end

        shadedErrorBar(centers_peakSpikeDensity,data_ripplePowerVsSpikeDensity(1,:),data_ripplePowerVsSpikeDensity(2,:),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        hold on, plot(centers_peakSpikeDensity,centers_peakSpikeDensity,'r--','linewidth',2), hold off
        xlabel('peak spike density'), ylabel('peak ripple power'), axis square, box on
        set(findobj(gcf,'type','axes'),'FontSize',12), set(gcf,'color','w')
        xlim([min(edges_peakSpikeDensity) max(edges_peakSpikeDensity)])


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %plot ripple power mean/peak as a function of elapsed time
        edges_elapsedTime = linspace(0,60*60,12); centers_elapsedTime = edges_elapsedTime(1:end-1)+mean(diff(edges_elapsedTime))/2;

        subplot(241)
        shadedErrorBar(centers_elapsedTime/(60),mean(data_ripplePower_elapsedTime{1},'omitnan'),std(data_ripplePower_elapsedTime{1},'omitnan')./sqrt(size(data_ripplePower_elapsedTime{1},1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        xlabel('elapsed time (min)'), ylabel('mean ripple power'), axis square

        subplot(242)
        shadedErrorBar(centers_elapsedTime/(60),mean(data_ripplePower_elapsedTime{2},'omitnan'),std(data_ripplePower_elapsedTime{2},'omitnan')./sqrt(size(data_ripplePower_elapsedTime{2},1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        xlabel('elapsed time (min)'), ylabel('peak ripple power'), axis square

        subplot(243)
        shadedErrorBar(centers_elapsedTime/(60),mean(data_spikeDensity_elapsedTime{1},'omitnan'),std(data_spikeDensity_elapsedTime{1},'omitnan')./sqrt(size(data_spikeDensity_elapsedTime{1},1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        xlabel('elapsed time (min)'), ylabel('mean spike density'), axis square

        subplot(244)
        shadedErrorBar(centers_elapsedTime/(60),mean(data_spikeDensity_elapsedTime{2},'omitnan'),std(data_spikeDensity_elapsedTime{2},'omitnan')./sqrt(size(data_spikeDensity_elapsedTime{2},1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        xlabel('elapsed time (min)'), ylabel('peak spike density'), axis square

        subplot(245), imagesc(data_ripplePower_elapsedTime{1}), axis square, xlabel('time'), ylabel('session'), set(gca,'xtick',[],'ytick',[]), %caxis([0 3])
        subplot(246), imagesc(data_ripplePower_elapsedTime{2}), axis square, xlabel('time'), ylabel('session'), set(gca,'xtick',[],'ytick',[]), %caxis([0 10])
        subplot(247), imagesc(data_spikeDensity_elapsedTime{1}), axis square, xlabel('time'), ylabel('session'), set(gca,'xtick',[],'ytick',[]), %caxis([0 3])
        subplot(248), imagesc(data_spikeDensity_elapsedTime{2}), axis square, xlabel('time'), ylabel('session'), set(gca,'xtick',[],'ytick',[]), %caxis([0 10])

        set(findobj(gcf,'type','axes'),'FontSize',12), set(gcf,'color','w')

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %plot difference in ripple power mean/peak for pairs of events as a function of time difference
        subplot(221)
        shadedErrorBar(centers_timesDiff,mean(data_ripplePower{1},'omitnan'),std(data_ripplePower{1},'omitnan')./sqrt(size(data_ripplePower{1},1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        hold on, shadedErrorBar(centers_timesDiff,mean(data_ripplePower{3},'omitnan'),std(data_ripplePower{3},'omitnan')./sqrt(size(data_ripplePower{1},1)),'lineprops',{'r','markerfacecolor','r','linewidth',2}), hold off
        xlabel('\Delta t (sec)'), ylabel('\Delta mean ripple power'), axis square

        subplot(222)
        shadedErrorBar(centers_timesDiff,mean(data_ripplePower{2},'omitnan'),std(data_ripplePower{2},'omitnan')./sqrt(size(data_ripplePower{2},1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        hold on, shadedErrorBar(centers_timesDiff,mean(data_ripplePower{4},'omitnan'),std(data_ripplePower{4},'omitnan')./sqrt(size(data_ripplePower{1},1)),'lineprops',{'r','markerfacecolor','r','linewidth',2}), hold off
        xlabel('\Delta t (sec)'), ylabel('\Delta peak ripple power'), axis square

        subplot(223)
        shadedErrorBar(centers_timesDiff,mean(data_spikeDensity{1},'omitnan'),std(data_spikeDensity{1},'omitnan')./sqrt(size(data_spikeDensity{1},1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        hold on, shadedErrorBar(centers_timesDiff,mean(data_spikeDensity{3},'omitnan'),std(data_spikeDensity{3},'omitnan')./sqrt(size(data_ripplePower{1},1)),'lineprops',{'r','markerfacecolor','r','linewidth',2}), hold off
        xlabel('\Delta t (sec)'), ylabel('\Delta mean spike density'), axis square

        subplot(224)
        shadedErrorBar(centers_timesDiff,mean(data_spikeDensity{2},'omitnan'),std(data_spikeDensity{2},'omitnan')./sqrt(size(data_spikeDensity{2},1)),'lineprops',{'k','markerfacecolor','k','linewidth',2})
        hold on, shadedErrorBar(centers_timesDiff,mean(data_spikeDensity{4},'omitnan'),std(data_spikeDensity{4},'omitnan')./sqrt(size(data_ripplePower{1},1)),'lineprops',{'r','markerfacecolor','r','linewidth',2}), hold off
        xlabel('\Delta t (sec)'), ylabel('\Delta peak spike density'), axis square
        set(findobj(gcf,'type','axes'),'FontSize',12), set(gcf,'color','w')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %plot inter-ripple interval
        histogram(replays_stats(:,15),linspace(0,2,24))
        xlabel('inter-ripple interval (s)'), ylabel('count'), axis square, set(gca,'fontsize',14), set(gcf,'color','w')
        xlim([0 1.5])
    

%%%%%

























                
                % %theta power from spike density
                %     [~,thetaPower_spikeDensity,~] = compute_filteredLFP(ThetaFreqRange,spikeDensity_smoothed,1/spikeDensityStepSize); 
                %     thetaPower_spikeDensity_smoothed = conv(thetaPower_spikeDensity,setUp_gaussFilt([1 10000],100/spikeDensityStepSize),'same');
                %     thetaPower_spikeDensity = [spikeDensity(:,1) thetaPower_spikeDensity_smoothed];
                % 
                %     runningTimes = [trialTimes(:,1) lickingTimes(:,1)];
                %     thetaPower_spikeDensity_runningOnly = compute_dataTemporalConcatenation(spikeDensity,runningTimes);
                %         mean_thetaPower_spikeDensity_runningOnly = mean(thetaPower_spikeDensity_runningOnly(:,end),'omitnan');
                %         std_thetaPower_spikeDensity_runningOnly = std(thetaPower_spikeDensity_runningOnly(:,end),'omitnan');
                %     thetaPower_spikeDensity(:,4) = (thetaPower_spikeDensity(:,3)-mean_thetaPower_spikeDensity_runningOnly)/std_thetaPower_spikeDensity_runningOnly;
