 clear all

Data = struct;
% cd analysis_RippleField/
% load replayEvents_rippleFieldStats_allCells
% cd ..

use_placeCells = 0;
use_computeTetrodeFields = 1;

numShuffles = 100;      
                
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
        load binDecoding_08
        load spikeDensity
            
        %histogram smoothing
            numSpatialBins_smoothing = 8;
            gaussFilt = setUp_gaussFilt(numSpatialBins,numSpatialBins_smoothing/binSize);

        %collect rate maps
            activityThr = 1;
            distThr = 4;%2; %proximity of spatial bins to barrier considered "overlapping the barriers"
            intactFracThr = 0.6; %fraction of field outside of barrier zone that must be passed
            barrierPotentialSmoothing = 12;
            binBehaviorSim_numBins_avg = 6;
            LBS_numBins_avg = 6;
            rateMap_sessions = cell(1,length(Run_Times));
            rateMap_coarse_sessions = cell(1,length(Run_Times));
            ind_cluster_sessions = cell(1,length(Run_Times));
            meanFiringRate_sessions = cell(1,length(Run_Times));
            meanSpatialFiringRate_sessions = cell(1,length(Run_Times));
            for sessionNum_decoder = 1:length(Run_Times)
                sessionNum = sessionNum_decoder;
                [ind_cluster,~,~,~,rateMap_smoothed_all,~,rateMap_smoothed_NaN_all,meanFiringRate_all,~,~,~,~,~,~,~,~,~,~,meanSpatialFiringRate_all,~,~,~,~,rateMap_coarse_smoothed_NaN_all] = load_rateMapsForDecoding_2D(clusters,sessionNum_decoder,Run_Times,numSpatialBins,meanFiringRateThr,spatialInfoThr,noiseOverlapThr,isolationThr,peakSnrThr,numSpatialBins_coarse);
                
                ind_cluster_sessions{sessionNum} = ind_cluster;
                rateMap_sessions{sessionNum} = rateMap_smoothed_NaN_all;
                rateMap_coarse_sessions{sessionNum} = rateMap_coarse_smoothed_NaN_all;
                meanFiringRate_sessions{sessionNum} = meanFiringRate_all;
                meanSpatialFiringRate_sessions{sessionNum} = meanSpatialFiringRate_all;
            end
        
            %find indices of suitable cells
                [ind_cluster_active,ind_cluster_active_pairSessions,ind_cluster_active_allSessions,ind_cluster_nonBarrierOverlapping,ind_cluster_nonBarrierOverlapping_pairSessions,ind_cluster_nonBarrierOverlapping_allSessions,ind_cluster_goodBehavioralSampling_pairSessions,cellBehaviorSim,binBehaviorSim,cellBarrierSim,binBarrierSim] = ...
                    load_ind_cluster_barrierMaze(clusters,Run_Times,ind_cluster_sessions,rateMap_sessions,rewardLocs,numSpatialBins,x_edges,y_edges,distThr,binSize,activityThr,intactFracThr,positions,speedThr,intactFracThr,barrierConfig_IDs,barrierPotentialSmoothing,binBehaviorSim_numBins_avg);

        %load trajectories and compute vector fields for rat trajectory
            circleRadius = 4; % radius of area (bins) for computing rat trajectory similarity
            vectorField_rat = cell(length(Run_Times),1);
            angleDistribution_rat = cell(length(Run_Times),1);
            trajectory_rat = cell(length(Run_Times),1);
            numDataPts_rat = cell(length(Run_Times),1);
            X_centers_rat = cell(length(Run_Times),1);
            for sessionNum_decoder = 1:length(Run_Times)
                sessionNum = sessionNum_decoder;
    
                %load rat trajectory (high speed)
                    positions_bins = compute_locsToBins(positions,x_edges,y_edges);
                    delx = positions_bins(2:end,2:3)-positions_bins(1:end-1,2:3);
                    x = [positions_bins [nan(1,2); delx]];
                    x(x(:,4)<speedThr,:) = nan;
                    jumps_sub = sqrt(diff(x(:,2)).^2+diff(x(:,3)).^2);
                    ind = find(jumps_sub>10);
                    x(ind,:) = NaN;
                    [x,trajectory_rat{sessionNum}] = compute_dataTemporalConcatenation(x, Run_Times{sessionNum});
                    delx = trajectory_rat{sessionNum}(:,end-1:end);
                    
                %compute vector field
                    [vectorField_rat{sessionNum},angleDistribution_rat{sessionNum},numDataPts_rat{sessionNum},X_centers_rat{sessionNum}] = compute_vectorField(trajectory_rat{sessionNum}(:,2:3),delx,circleRadius,numSpatialBins);
            end
                    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
    %iterate through sessions
    for sessionNum_decoder = 1:length(Run_Times)
        sessionNum = sessionNum_decoder;
        
        Times = load_timeBounds(Run_Times{sessionNum});
        
        %load immobility times
            positions_session = compute_dataTemporalConcatenation(positions,load_timeBounds(Run_Times{sessionNum}));
            x = positions_session(:,4); x(positions_session(:,4)>speedThr) = nan;
            [boundaries,lengths] = compute_allSequences_NaNseparated(x);        
            immobilityTimes = [positions_session(boundaries(:,1),1) positions_session(boundaries(:,2),1)];

        %load barriers
            barrierPotentialSmoothing = 12;
            barrierConfigs_allBarriers = nchoosek(1:12,6);
            numConfigs = size(barrierConfigs_allBarriers,1);
            rewardLocs_scaled = compute_locsToBins(rewardLocs{sessionNum},x_edges,y_edges);
            [~,barriers_mat,~,~,~,walls_mat] = load_idealizedBarrierLocations(rewardLocs_scaled,barrierConfig_IDs{sessionNum});
            smoothingFilt = setUp_gaussFilt([100 100],barrierPotentialSmoothing);

        %%%%%%%%%%%%%%%
        %%%rate maps%%%
        %%%%%%%%%%%%%%%
            
            [ind_cluster,rateMap_smoothed,rateMap,rateMap_smoothed_NaN,rateMap_smoothed_all,~,rateMap_smoothed_NaN_all,meanFiringRate,spatialInfo,noiseOverlap,isolation,peakSnr,~,~,~,~,stability,~,~,~,~,~,~,~,tetrode,ind_cluster_interneurons,ind_cluster_all] = load_rateMapsForDecoding_2D(clusters,sessionNum_decoder,Run_Times,numSpatialBins,meanFiringRateThr,spatialInfoThr,noiseOverlapThr,isolationThr,peakSnrThr,numSpatialBins_coarse);
            if use_placeCells==1
                rateMapSum_smoothed_NaN = reshape(nansum(rateMap_smoothed_NaN_all(ind_cluster,:),1),numSpatialBins); rateMapSum_smoothed_NaN(rateMapSum_smoothed_NaN==0) = nan;
            else
                rateMapSum_smoothed_NaN = reshape(nansum(rateMap_smoothed_NaN_all(ind_cluster_all,:),1),numSpatialBins); rateMapSum_smoothed_NaN(rateMapSum_smoothed_NaN==0) = nan;
            end

        %%%%%%%%%%%%%    
        %%%Replays%%%
        %%%%%%%%%%%%%
        
            %load replays
                replay_dispersionThr = 12; 
                load_ind_replayEvents

                [replays,replays_NaN,replays_NaNremoved,replays_interp,~,~,~,~,replays_smoothed] = compute_replaysConcatenation(decoder_replay(sessionNum),ind_replay);
                replays = replays_NaNremoved;
                replays_NaNremoved = replays; replays_NaNremoved(isnan(replays_NaNremoved(:,2)),:) = [];
                    ind_NaN = find(isnan(replays(:,2)));
                replayTimes = timePoints(ind_replay,:);
                replayTimes_old = timePoints(ind_replay_old,:);

            %replay position density
                replays_locs = compute_binsToLocs(replays(:,2:3),x_edges,y_edges);
                replayDensity_mat = hist3([replays_locs(:,2) replays_locs(:,1)],'Edges',{y_edges x_edges}); replayDensity_mat = replayDensity_mat(1:end-1,1:end-1);
                [replayDensity_mat_smoothed,replayDensity_mat_smoothed_NaN] = compute_smoothedPositionHist(replayDensity_mat,gaussFilt,replayDensity_mat);
                
            %load LFP
                name = strcat('LFP_',num2str(sessionNum),'.mat');
                if exist(name,'file')==0, continue, end
                load(name)

            %ripple power
                ripplePower = LFP_ripplePower_mean;
                ripplePower_session = compute_dataTemporalConcatenation(ripplePower,load_timeBounds(Run_Times{sessionNum}));

                ripplePower_immobility = compute_dataTemporalConcatenation(ripplePower_session,immobilityTimes);
                    mean_ripplePower = mean(ripplePower_immobility(:,end),'omitnan');
                    std_ripplePower = std(ripplePower_immobility(:,end),'omitnan');
                ripplePower_session(:,3) = (ripplePower_session(:,2)-mean_ripplePower)/std_ripplePower;

                ripplePower_replayOnly = compute_dataTemporalConcatenation(ripplePower_session,replayTimes);

                replays_LFP = compute_dataInterpolation(ripplePower_session,replays(:,1));
                replays_LFP_NaNremoved = replays_LFP; replays_LFP_NaNremoved(ind_NaN,:) = [];

                %compute fields
                    [replays_mat_smoothed_LFP,replays_mat_smoothed_NaN_LFP,~,replays_mat_raw_smoothed_NaN_LFP] = compute_ampField(replays(:,2:3),replays_LFP(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
                    [replays_mat_smoothed_LFP_nonZscored,replays_mat_smoothed_NaN_LFP_nonZscored,~,replays_mat_raw_smoothed_NaN_LFP_nonZscored] = compute_ampField(replays(:,2:3),replays_LFP(:,end-1),replayDensity_mat,numSpatialBins,gaussFilt);


                        % subplot(131), imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
                        % subplot(132), imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
                        % for i = 5:24%length(ind_replay)
                        %     [replay,replay_NaN,replay_NaNremoved,replay_interp] = compute_replaysConcatenation(decoder_replay(sessionNum),ind_replay(i));
                        %     replay_LFP = compute_dataInterpolation(ripplePower_session,replay_NaNremoved(:,1));
                        % 
                        %     replayStart = replay_NaNremoved(1,2:3);
                        %     dist = sqrt((replayStart(1)-ratLoc(ind_replay(i),1))^2 + (replayStart(2)-ratLoc(ind_replay(i),2))^2);
                        %     if dist < 10, subplot(131)
                        %     else, subplot(132)
                        %     end
                        %     hold on, plot(replay_NaNremoved(:,2),replay_NaNremoved(:,3),'k','linewidth',6), hold off
                        %     hold on, cplot(replay_NaNremoved(:,2),replay_NaNremoved(:,3),replay_LFP(:,end),'linewidth',4), hold off
                        %     hold on, plot(replay_NaNremoved(1,2),replay_NaNremoved(1,3),'g.','markersize',16), hold off
                        %     hold on, plot(ratLoc(ind_replay(i),1),ratLoc(ind_replay(i),2),'r.','markersize',16), hold off
                        %     plot_mazeProperties
                        %     box on, caxis([0 2.5])
                        % end
                        % 
                        % subplot(133)
                        % imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
                        % hold on, imagesc(replays_mat_smoothed_NaN_LFP,'AlphaData',~isnan(replays_mat_smoothed_NaN_LFP)), hold off
                        % plot_mazeProperties
                        % caxis([0 2])
                        % 
                        % keyboard

                %ripple events
                    eventZscoreThr = 2;
                    eventDurationThr = 0.015;
                    eventMergeThr = 0;
                    rippleEvents = compute_events_times([ripplePower_replayOnly(:,1),ripplePower_replayOnly(:,end)],eventZscoreThr,eventMergeThr,eventDurationThr);
                    rippleTimes = rippleEvents(:,3);

                %directionality: replays w/ vs. w/o ripples 
                    %compute field boundary
                        search_dist = 3;
                        search_minPts = 250;
                        num_reSamples = 10000;
                        thr = 0.3;
                        [S_fields,S_peakLocs,S_COMs,spkPos_samples,idx,S_numPts,S_fieldBoundary] = compute_clusters_densityBasedClusters(replays_mat_smoothed_LFP,num_reSamples,search_dist,search_minPts,numSpatialBins,spatialDim,numSpatialBins_smoothing,binSize,thr);
                        rippleFieldLocs = S_peakLocs;
                        rippleFieldBoundaries = S_fieldBoundary;

                        while isempty(rippleFieldBoundaries)
                            [S_fields,S_peakLocs,S_COMs,spkPos_samples,idx,S_numPts,S_fieldBoundary] = compute_clusters_densityBasedClusters(replays_mat_smoothed_LFP,num_reSamples,search_dist,search_minPts,numSpatialBins,spatialDim,numSpatialBins_smoothing,binSize,thr);
                            rippleFieldLocs = S_peakLocs;
                            rippleFieldBoundaries = S_fieldBoundary;
                        end
                        ind = find(S_numPts>1000);

                        rippleFieldLocs = rippleFieldLocs(ind,:);
                        rippleFieldBoundaries = rippleFieldBoundaries(ind,:);

                    %shuffle ripple times
                        rippleTimes_shuffle = cell(1,numShuffles);
                        for i = 1:numShuffles

                            randShift = round(rand*size(ripplePower_replayOnly,1));
                            ripplePower_replayOnly_shuffle = [ripplePower_replayOnly(:,1) circshift(ripplePower_replayOnly(:,2:end),randShift)];

                            %ripple times
                                rippleEvents_shuffle = compute_events_times([ripplePower_replayOnly_shuffle(:,1),ripplePower_replayOnly_shuffle(:,3)],eventZscoreThr,eventMergeThr,eventDurationThr);
                                rippleTimes_shuffle{i} = rippleEvents_shuffle(:,3);
                        end

                    %compute ripple field passes
                        rippleFieldPasses_pvalue = nan(length(rippleFieldBoundaries),1);
                        rippleFieldPasses_numPasses = nan(length(rippleFieldBoundaries),1);
                        rippleFieldPasses_mvl = nan(length(rippleFieldBoundaries),1);
                        rippleFieldPasses_mvl_shuffles = nan(length(rippleFieldBoundaries),numShuffles);
                        for i = 1:length(rippleFieldBoundaries)
                            ind_withinField = find(inpolygon(replays(:,2),replays(:,3),rippleFieldBoundaries{i}(:,1),rippleFieldBoundaries{i}(:,2)));    
                            passes_bins_NaNs = nan(size(replays)); passes_bins_NaNs(ind_withinField,:) = replays(ind_withinField,:);

                            binSize = 2;
                            sequence_delxThr = 20;
                            sequence_deltThr = 0.05;
                            sequence_durationThr = 0.1;
                            shiftSizeDecoding = 0.005;
                            [boundaries,lengths] = compute_allSequences_NaNseparated(passes_bins_NaNs(:,2)); %find sequences bounded by NaNs       
                            [boundaries,lengths] = compute_allSequences_NaNseparated_merge(passes_bins_NaNs,boundaries,sequence_delxThr,sequence_deltThr,binSize); %merge sequences if gap is delx<delxThr and delt<deltThr 
                            timeBounds = [replays(boundaries(:,1),1) replays(boundaries(:,2),1)];
                            numPasses = size(boundaries,1);

                            pass_angle = nan(numPasses,1);
                            pass_wRipple = nan(numPasses,1);
                            pass_wRipple_shuffle = nan(numPasses,numShuffles);
                            pass_numPts = nan(numPasses,1);
                            for j = 1:numPasses
                                replay = compute_dataTemporalConcatenation(replays,timeBounds(j,:));
                                replay = unique(replay(:,2:3),'rows');
                                pass_numPts(j) = size(replay,1);
                                % if compute_sequenceDistance(replay)<15, continue, end
                                if size(replay,1)<5, continue, end

                                pass_angle(j) = circ_mean(atan2(diff(replay(:,2)),diff(replay(:,1))));
                                pass_wRipple(j) = length(compute_dataTemporalConcatenation(rippleTimes,timeBounds(j,:)));
                                for k = 1:numShuffles
                                    pass_wRipple_shuffle(j,k) = length(compute_dataTemporalConcatenation(rippleTimes_shuffle{k},timeBounds(j,:)));
                                end

                                % subplot(121)
                                % plot(replays(:,2),replays(:,3),'k')
                                % hold on, plot(fieldBoundary{i}(:,1),fieldBoundary{i}(:,2),'g'), hold off
                                % hold on, plot(passes_bins_NaNs(:,2),passes_bins_NaNs(:,3),'b'), hold off
                                % hold on, cplot(replay(:,1),replay(:,2),1:size(replay,1),'linewidth',4), hold off
                                % 
                                % subplot(122)
                                % hist(atan2(diff(replay(:,2)),diff(replay(:,1))))
                            end

                            %compute mean vector length of angle distributions
                                angles = linspace(-pi,pi,18); angles_centers = angles(1:end-1) + mean(diff(angles))/2;
                                h = histcounts(pass_angle(pass_wRipple>0),angles,'normalization','probability');
                                [~,mvl] = compute_meanVectorLength(h,angles_centers);

                                mvl_shuffles = nan(1,numShuffles);
                                for j = 1:numShuffles
                                    h = histcounts(pass_angle(pass_wRipple_shuffle(:,j)>0),angles,'normalization','probability');
                                    [~,mvl_shuffles(j)] = compute_meanVectorLength(h,angles_centers);
                                    % plot(angles_centers,h), keyboard
                                end

                            %compute pvalue
                                pvalue = (1+nansum(mvl_shuffles>mvl))/(1+numShuffles);

                            rippleFieldPasses_pvalue(i) = pvalue;
                            rippleFieldPasses_numPasses(i) = numPasses;
                            rippleFieldPasses_mvl(i) = mvl;
                            rippleFieldPasses_mvl_shuffles(i,:) = mvl_shuffles;
                        end  

                %ripple timing shift
                    shifts = -0.5:0.05:0.5;
                    spatialInfo_LFP_timeShift = nan(1,length(shifts));
                    for i = 1:length(shifts)
                        ripplePower_session_shift = ripplePower_session;
                        ripplePower_session_shift(:,1) = ripplePower_session_shift(:,1)+shifts(i);
                        replays_LFP_shift = compute_dataInterpolation(ripplePower_session_shift,replays(:,1));
                        [replays_mat_smoothed_LFP_shift_nonZscored,replays_mat_smoothed_NaN_LFP_shift_nonZscored] = compute_ampField(replays(:,2:3),replays_LFP_shift(:,end-1),replayDensity_mat,numSpatialBins,gaussFilt);

                        spatialInfo_LFP_timeShift(i) = compute_spatialInfo(replays_mat_smoothed_NaN_LFP_shift_nonZscored,replayDensity_mat);
                    end

            %spike density
                if use_placeCells == 1
                    spikeDensity_session = load_spikeDensity_placeCells(clusters,ind_cluster,spikeDensityStepSize,load_timeBounds(Run_Times{sessionNum}));    
                else
                    spikeDensity_session = load_spikeDensity_placeCells(clusters,ind_cluster_all,spikeDensityStepSize);                         
                end
                spikeDensity_smoothed = conv(spikeDensity_session(:,2),setUp_gaussFilt([1 1000],0.1/spikeDensityStepSize),'same');
                spikeDensity_session = [spikeDensity_session(:,1) spikeDensity_session(:,2) spikeDensity_smoothed];

                spikeDensity_immobility = compute_dataTemporalConcatenation(spikeDensity_session,immobilityTimes);
                    mean_spikeDensity = mean(spikeDensity_immobility(:,end),'omitnan');
                    std_spikeDensity = std(spikeDensity_immobility(:,end),'omitnan');
                spikeDensity_session(:,4) = (spikeDensity_session(:,3)-mean_spikeDensity)/std_spikeDensity;

                replays_spikeDensity = compute_dataInterpolation(spikeDensity_session,replays(:,1));
                replays_spikeDensity_NaNremoved = replays_spikeDensity; replays_spikeDensity_NaNremoved(ind_NaN,:) = [];

                %compute field
                [replays_mat_smoothed_spikeDensity,replays_mat_smoothed_NaN_spikeDensity,~,replays_mat_raw_smoothed_NaN_spikeDensity] = compute_ampField(replays(:,2:3),replays_spikeDensity(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
                [replays_mat_smoothed_spikeDensity_nonZscored,replays_mat_smoothed_NaN_spikeDensity_nonZscored,~,replays_mat_raw_smoothed_NaN_spikeDensity_nonZscored] = compute_ampField(replays(:,2:3),replays_spikeDensity(:,end-1),replayDensity_mat,numSpatialBins,gaussFilt);

            %Tetrodes: ripple and spike density fields
                replays_mat_smoothed_NaN_LFP_tets = nan(64,prod(numSpatialBins));
                replays_mat_smoothed_NaN_spikeDensity_tets = nan(64,prod(numSpatialBins));
                tets_numCells = nan(64,1);
                tets_ripplePower = nan(64,1);      

                if ~isempty(dir('*merged.LFP')), name = dir('*merged.LFP');
                elseif ~isempty(dir('*merge.LFP')), name = dir('*merge.LFP');
                end
                cd(name.name), LFPs = dir('*.mat'); cd ..

                times_LFP_replayOnly = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),replayTimes_old);
                times_LFP_immobility = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),immobilityTimes);
                if length(times_LFP_immobility) ~= size(LFP_immobility_ripplePower,2)
                    immobilityTimes(diff(immobilityTimes,[],2)<1,:) = []; 
                    times_LFP_immobility = compute_dataTemporalConcatenation(LFP_ripplePower_mean(:,1:2),immobilityTimes);
                end

                ripplePower_tets_replayOnly = [times_LFP_replayOnly(:,1) LFP_replay_ripplePower'];
                    ripplePower_tets_replayOnly = compute_dataTemporalConcatenation(ripplePower_tets_replayOnly,replayTimes);
                ripplePower_tets_immobility = [times_LFP_immobility(:,1) LFP_immobility_ripplePower'];
                    mean_ripplePower_tets = mean(ripplePower_tets_immobility(:,2:end),'omitnan');
                    std_ripplePower_tets = std(ripplePower_tets_immobility(:,2:end),'omitnan');
                ripplePower_tets_replayOnly(:,2:end) = (ripplePower_tets_replayOnly(:,2:end)-repmat(mean_ripplePower_tets,size(ripplePower_tets_replayOnly,1),1))./repmat(std_ripplePower_tets,size(ripplePower_tets_replayOnly,1),1);
                ripplePower_tets_immobility(:,2:end) = (ripplePower_tets_immobility(:,2:end)-repmat(mean_ripplePower_tets,size(ripplePower_tets_immobility,1),1))./repmat(std_ripplePower_tets,size(ripplePower_tets_immobility,1),1);

                for i = 1:64
                    if use_placeCells == 1
                        ind_cluster_sub = ind_cluster(tetrode(ind_cluster)==i);
                    else
                        ind_cluster_sub = ind_cluster_all(tetrode(ind_cluster_all)==i);
                    end
                    if length(ind_cluster_sub)>5 %only look at tetrodes with at least 5 place cells

                        %ripple power
                            replays_LFP_tet = compute_dataInterpolation([ripplePower_tets_replayOnly(:,1) ripplePower_tets_replayOnly(:,i+1)],replays(:,1));
                            [replays_mat_smoothed_LFP_tet,replays_mat_smoothed_NaN_LFP_tet] = compute_ampField(replays(:,2:3),replays_LFP_tet(:,end),replayDensity_mat,numSpatialBins,gaussFilt);

                            cd(name.name)
                            load(LFPs(i).name)
                            LFP = LFP_Data; clear LFP_Data
                            cd ..

                            LFP = compute_dataTemporalConcatenation(LFP,load_timeBounds(Run_Times{sessionNum}));
                            [~,ind_unique,~] = unique(LFP(:,1));
                            LFP = LFP(ind_unique,:);
                            [~,LFP_filt_ripplePower,LFP_filt_ripplePhase] = compute_filteredLFP(SWRFreqRange,LFP(:,2),LFPSampRate); 

                            ripplePower_tet = [LFP(:,1) LFP_filt_ripplePower];
                            ripplePower_tet_smoothed = conv(ripplePower_tet(:,2),setUp_gaussFilt([1 1000],100),'same');
                            ripplePower_tet = [ripplePower_tet ripplePower_tet_smoothed];

                            immobility_LFP_tet = compute_dataTemporalConcatenation(ripplePower_tet,immobilityTimes);
                                mean_LFP = nanmean(immobility_LFP_tet(:,2));
                                std_LFP = nanstd(immobility_LFP_tet(:,2));
                            ripplePower_tet(:,3) = (ripplePower_tet(:,2)-mean_LFP)/std_LFP;

                            replays_LFP_tet = compute_dataInterpolation(ripplePower_tet,replays(:,1));

                        %spike density
                            spikeDensity_tet = load_spikeDensity_placeCells(clusters,ind_cluster_sub,spikeDensityStepSize);  
                            spikeDensity_tet_smoothed = conv(spikeDensity_tet(:,2),setUp_gaussFilt([1 1000],0.1/spikeDensityStepSize),'same');
                            spikeDensity_tet = [spikeDensity_tet spikeDensity_tet_smoothed];

                            immobility_spikeDensity_tet = compute_dataTemporalConcatenation(spikeDensity_tet,immobilityTimes);
                                mean_spikeDensity = nanmean(immobility_spikeDensity_tet(:,end));
                                std_spikeDensity = nanstd(immobility_spikeDensity_tet(:,end));
                            spikeDensity_tet(:,4) = (spikeDensity_tet(:,3) - mean_spikeDensity)/std_spikeDensity;

                            replays_spikeDensity_tet = compute_dataInterpolation(spikeDensity_tet,replays(:,1));
                            [replays_mat_smoothed_spikeDensity_tet,replays_mat_smoothed_NaN_spikeDensity_tet] = compute_ampField(replays(:,2:3),replays_spikeDensity_tet(:,end),replayDensity_mat,numSpatialBins,gaussFilt);

                        %save
                            replays_mat_smoothed_NaN_LFP_tets(i,:) = replays_mat_smoothed_NaN_LFP_tet(:);
                            replays_mat_smoothed_NaN_spikeDensity_tets(i,:) = replays_mat_smoothed_NaN_spikeDensity_tet(:);
                            tets_numCells(i) = length(ind_cluster_sub);
                            tets_ripplePower(i) = nanmean(replays_LFP_tet(:,end));

                    end
                end

                % Data(ii).tets_ripplePower = tets_ripplePower;
                % Data(ii).tets_numCells = tets_numCells;
                % Data(ii).LFP.replays_amp_mat_tets = replays_mat_smoothed_NaN_LFP_tets;
                % Data(ii).spikeDensity.replays_amp_mat_tets = replays_mat_smoothed_NaN_spikeDensity_tets;
                % ii = ii+1; continue

                        
        %%%%%%%%%%%%%%%%%%
        %%%rat position%%%
        %%%%%%%%%%%%%%%%%%
        
            positions_sub = compute_dataTemporalConcatenation(positions,Run_Times{sessionNum});
            positions_sub(positions_sub(:,4)<2*speedThr,:) = nan;
            positionDensity_mat = hist3([positions_sub(:,3) positions_sub(:,2)],'Edges',{y_edges x_edges}); positionDensity_mat = positionDensity_mat(1:end-1,1:end-1);
            [positionDensity_mat_smoothed,positionDensity_mat_smoothed_NaN] = compute_smoothedPositionHist(positionDensity_mat,gaussFilt,positionDensity_mat);
            
            % positions_immobility_sub = compute_dataTemporalConcatenation(positions,Run_Times{sessionNum});
            % positions_immobility_sub(positions_immobility_sub(:,4)>speedThr,:) = nan;
            % positionDensity_immobility_mat = hist3([positions_immobility_sub(:,3) positions_immobility_sub(:,2)],'Edges',{y_edges x_edges}); positionDensity_immobility_mat = positionDensity_immobility_mat(1:end-1,1:end-1);
            % [~,positionDensity_immobility_mat_smoothed_NaN] = compute_smoothedPositionHist(positionDensity_immobility_mat,gaussFilt,positionDensity_immobility_mat);

            positions_replay = compute_dataInterpolation(positions,replays(:,1));
            positionDensity_replays_mat = hist3([positions_replay(:,3) positions_replay(:,2)],'Edges',{y_edges x_edges}); positionDensity_replays_mat = positionDensity_replays_mat(1:end-1,1:end-1);
            [~,positionDensity_replays_mat_smoothed] = compute_smoothedPositionHist(positionDensity_replays_mat,gaussFilt,positionDensity_replays_mat);
            positionDensity_immobility_mat_smoothed_NaN = positionDensity_replays_mat_smoothed; %!!!!!!!!!!!! "immobility" being redefined to where the rat was during replay, which is more pertinent to the discussion

        %%%%%%%%%%%%%%%%%%%%%
        %%%Across sessions%%%
        %%%%%%%%%%%%%%%%%%%%%
        
            if sessionNum>1
                ind_cluster_unstable = find(stability(:,sessionNum-1)==0);
                ind_cluster_stable = find(stability(:,sessionNum-1)==1);
                rateMapSum_unstable_smoothed_NaN = reshape(sum(rateMap_smoothed_NaN_all(ind_cluster_unstable,:)),numSpatialBins);
                rateMapSum_stable_smoothed_NaN = reshape(sum(rateMap_smoothed_NaN_all(ind_cluster_stable,:)),numSpatialBins);
                localBarrierSim = compute_localBarrierSimilarity(rewardLocs,numSpatialBins,x_edges,y_edges,barrierConfig_IDs,barrierPotentialSmoothing,LBS_numBins_avg,binSize,[sessionNum-1 sessionNum]);
                behaviorSim = compute_PVCorrelation(angleDistribution_rat{sessionNum-1},angleDistribution_rat{sessionNum},numSpatialBins);

                barrierDiff = -localBarrierSim;
                behaviorDiff = -behaviorSim;

                if length(unique(barrierDiff))==1, barrierDiff = nan(numSpatialBins); end
                if length(unique(behaviorDiff))==1, behaviorDiff = nan(numSpatialBins); end
            else
                rateMapSum_unstable_smoothed_NaN = nan(numSpatialBins);
                rateMapSum_stable_smoothed_NaN = nan(numSpatialBins);
                barrierDiff = nan(numSpatialBins);
                behaviorDiff = nan(numSpatialBins);
            end            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %spatial info
            spatialInfo_LFP = compute_spatialInfo(replays_mat_smoothed_NaN_LFP_nonZscored,replayDensity_mat);
            spatialInfo_spikeDensity = compute_spatialInfo(replays_mat_smoothed_NaN_spikeDensity_nonZscored,replayDensity_mat); 

        %ripple vs spike density field
            LFPvsSpikeDensityCorr = compute_corrCoef([replays_mat_smoothed_NaN_LFP(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

        %rat vs replay density
            replayRatDensityCorr = compute_corrCoef([positionDensity_mat_smoothed_NaN(:) replayDensity_mat_smoothed_NaN(:)]);

        %place field correlation
            rateMapSumCorr_LFP = compute_corrCoef([rateMapSum_smoothed_NaN(:) replays_mat_smoothed_NaN_LFP(:)]);
            rateMapSumCorr_spikeDensity = compute_corrCoef([rateMapSum_smoothed_NaN(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

        %replay trajectory correlation
            replayTrajCorr_LFP = compute_corrCoef([replayDensity_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_LFP(:)]);
            replayTrajCorr_spikeDensity = compute_corrCoef([replayDensity_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

        %rat trajectory correlation
            ratTrajCorr_LFP = compute_corrCoef([positionDensity_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_LFP(:)]);
            ratTrajCorr_spikeDensity = compute_corrCoef([positionDensity_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

            ratTrajCorr_immobility_LFP = compute_corrCoef([positionDensity_immobility_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_LFP(:)]);
            ratTrajCorr_immobility_spikeDensity = compute_corrCoef([positionDensity_immobility_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

        %rat trajectory meanVectorLength(MVL)
            ratMVL = reshape(1-vectorField_rat{sessionNum}(:,1),numSpatialBins);
            ratMVLCorr_LFP = compute_corrCoef([ratMVL(:) replays_mat_smoothed_NaN_LFP(:)]);
            ratMVLCorr_spikeDensity = compute_corrCoef([vectorField_rat{sessionNum}(:,1) replays_mat_smoothed_NaN_spikeDensity(:)]);

        %proximity to nearest reward locs
            [X,Y] = meshgrid(1:numSpatialBins(2),1:numSpatialBins(1)); X = [X(:) Y(:)]; 
            nearestLoc = compute_nearestLoc(X,compute_locsToBins(rewardLocs{sessionNum},x_edges,y_edges));
            distToRewardLocs_mat = -reshape(nearestLoc(:,4),numSpatialBins);
            proxToNearestRewardCorr_LFP = compute_corrCoef([distToRewardLocs_mat(:) replays_mat_smoothed_NaN_LFP(:)]);
            proxToNearestRewardCorr_spikeDensity = compute_corrCoef([distToRewardLocs_mat(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

        %distance to nearest walls/barriers
            [X,Y] = meshgrid(1:numSpatialBins(2),1:numSpatialBins(1)); X = [X(:) Y(:)]; 
            nearestLoc = compute_nearestLoc(X,[barriers_mat; walls_mat]);
            distToNearestBarrier_mat = reshape(nearestLoc(:,4),numSpatialBins);
            distToNearestBarrierCorr_LFP = compute_corrCoef([distToNearestBarrier_mat(:) replays_mat_smoothed_NaN_LFP(:)]);
            distToNearestBarrierCorr_spikeDensity = compute_corrCoef([distToNearestBarrier_mat(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

        %across session measures
            if sessionNum>1
                %place field correlation
                    rateMapSumCorr_LFP_unstable = compute_corrCoef([rateMapSum_unstable_smoothed_NaN(:) replays_mat_smoothed_NaN_LFP(:)]);
                    rateMapSumCorr_LFP_stable = compute_corrCoef([rateMapSum_stable_smoothed_NaN(:) replays_mat_smoothed_NaN_LFP(:)]);
                    rateMapSumCorr_spikeDensity_unstable = compute_corrCoef([rateMapSum_unstable_smoothed_NaN(:) replays_mat_smoothed_NaN_spikeDensity(:)]);
                    rateMapSumCorr_spikeDensity_stable = compute_corrCoef([rateMapSum_stable_smoothed_NaN(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

                %barrierDiff correlation
                    barrierDiffCorr_LFP = compute_corrCoef([barrierDiff(:) replays_mat_smoothed_NaN_LFP(:)]);
                    barrierDiffCorr_spikeDensity = compute_corrCoef([barrierDiff(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

                %behavioral similarity correlation
                    behaviorDiffCorr_LFP = compute_corrCoef([behaviorDiff(:) replays_mat_smoothed_NaN_LFP(:)]);
                    behaviorDiffCorr_spikeDensity = compute_corrCoef([behaviorDiff(:) replays_mat_smoothed_NaN_spikeDensity(:)]);

            else
                rateMapSumCorr_LFP_unstable = nan;
                rateMapSumCorr_LFP_stable = nan;
                rateMapSumCorr_spikeDensity_unstable = nan;
                rateMapSumCorr_spikeDensity_stable = nan;
                barrierDiffCorr_LFP = nan;
                barrierDiffCorr_spikeDensity = nan;
                behaviorDiffCorr_LFP = nan;
                behaviorDiffCorr_spikeDensity = nan;
            end

        %stability: 1st half vs 2nd half & interleaved
            for tt = 1:2
                spikeDensity_mat_vec = nan(prod(numSpatialBins),2);
                LFP_mat_vec = nan(prod(numSpatialBins),2);
                replayDensity_mat_vec = nan(prod(numSpatialBins),2);
                for t = 1:2
                    if tt==1 %halves
                        if t==1 %halves
                            edges_sub = timePoints(ind_replay(1:round(length(ind_replay)/2)),:);
                        elseif t==2
                            edges_sub = timePoints(ind_replay(round(length(ind_replay)/2)+1:end),:);
                        end
                    else
                        if t==1 %interleaved
                            edges_sub = timePoints(ind_replay(1:2:end),:);
                        elseif t==2
                            edges_sub = timePoints(ind_replay(2:2:end),:);
                        end
                    end

                    %replays
                        [~,replays_sub] = compute_dataTemporalConcatenation(replays,edges_sub);
                        replays_locs_sub = compute_binsToLocs(replays_sub(:,2:3),x_edges,y_edges);
                        replayDensity_mat_sub = hist3([replays_locs_sub(:,2) replays_locs_sub(:,1)],'Edges',{y_edges x_edges}); replayDensity_mat_sub = replayDensity_mat_sub(1:end-1,1:end-1);
                        replayDensity_mat_vec(:,t) = replayDensity_mat_sub(:);

                    %SWR
                        replays_LFP_sub = compute_dataInterpolation(replays_LFP_NaNremoved,replays_sub(:,1));
                        [replays_mat_smoothed_LFP_sub,replays_mat_smoothed_NaN_LFP_sub,~,replays_mat_raw_smoothed_NaN_LFP_sub] = compute_ampField(replays_sub(:,2:3),replays_LFP_sub(:,end),replayDensity_mat_sub,numSpatialBins,gaussFilt);
                        LFP_mat_vec(:,t) = replays_mat_smoothed_NaN_LFP_sub(:);

                    %spike density
                        replays_spikeDensity_sub = compute_dataInterpolation(replays_spikeDensity_NaNremoved,replays_sub(:,1));
                        [replays_mat_smoothed_spikeDensity_sub,replays_mat_smoothed_NaN_spikeDensity_sub,~,replays_mat_raw_smoothed_NaN_spikeDensity_sub] = compute_ampField(replays_sub(:,2:3),replays_spikeDensity_sub(:,end),replayDensity_mat_sub,numSpatialBins,gaussFilt);
                        spikeDensity_mat_vec(:,t) = replays_mat_smoothed_NaN_spikeDensity_sub(:);

                    %load
                    if tt==1
                        Data(ii).LFP.stability_halves.window(t).replay = replays_sub;
                        Data(ii).LFP.stability_halves.window(t).replays_amp = replays_LFP_sub;
                        Data(ii).LFP.stability_halves.window(t).replays_amp_mat = replays_mat_smoothed_NaN_LFP_sub;

                        Data(ii).spikeDensity.stability_halves.window(t).replay = replays_sub;
                        Data(ii).spikeDensity.stability_halves.window(t).replays_amp = replays_spikeDensity_sub;
                        Data(ii).spikeDensity.stability_halves.window(t).replays_amp_mat = replays_mat_smoothed_NaN_spikeDensity_sub;
                    else
                        Data(ii).LFP.stability_interleaved.window(t).replay = replays_sub;
                        Data(ii).LFP.stability_interleaved.window(t).replays_amp = replays_LFP_sub;
                        Data(ii).LFP.stability_interleaved.window(t).replays_amp_mat = replays_mat_smoothed_NaN_LFP_sub;

                        Data(ii).spikeDensity.stability_interleaved.window(t).replay = replays_sub;
                        Data(ii).spikeDensity.stability_interleaved.window(t).replays_amp = replays_spikeDensity_sub;
                        Data(ii).spikeDensity.stability_interleaved.window(t).replays_amp_mat = replays_mat_smoothed_NaN_spikeDensity_sub;
                    end

                    % %plot
                    %     subplot(2,2,t)
                    %     imagesc(zeros(size(replayDensity_mat_sub))), caxis([0 3])
                    %     hold on, plot(replays_sub(:,2),replays_sub(:,3),'k','linewidth',2), hold off
                    %     hold on, cplot(replays_sub(:,2),replays_sub(:,3),replays_LFP_sub(:,end),'linewidth',1), hold off
                    %     plot_mazeProperties
                    % 
                    %     subplot(2,2,2+t)
                    %     imagesc(compute_zscore_squareMatrix(replays_mat_smoothed_NaN_LFP_sub)), plot_mazeProperties, caxis([-0.5 2])
                    %     plot_mazeProperties

                end
                if tt==1
                    stability_halves_LFP = compute_corrCoef(LFP_mat_vec);
                    stability_halves_spikeDensity = compute_corrCoef(spikeDensity_mat_vec);
                    stability_halves_replayDensity = compute_corrCoef(replayDensity_mat_vec);
                else
                    stability_interleaved_LFP = compute_corrCoef(LFP_mat_vec);
                    stability_interleaved_spikeDensity = compute_corrCoef(spikeDensity_mat_vec);
                    stability_interleaved_replayDensity = compute_corrCoef(replayDensity_mat_vec);
                end
            end            

        %shuffles
            LFPvsSpikeDensityCorr_shuffle = nan(1,numShuffles);
            spatialInfo_shuffle_LFP = nan(1,numShuffles);
            spatialInfo_shuffle_spikeDensity = nan(1,numShuffles);
            rateMapSumCorr_shuffle_LFP = nan(1,numShuffles);
            rateMapSumCorr_shuffle_spikeDensity = nan(1,numShuffles);
            rateMapSumCorr_shuffle_LFP_unstable = nan(1,numShuffles);
            rateMapSumCorr_shuffle_LFP_stable = nan(1,numShuffles);
            rateMapSumCorr_shuffle_spikeDensity_unstable = nan(1,numShuffles);
            rateMapSumCorr_shuffle_spikeDensity_stable = nan(1,numShuffles);
            stability_halves_shuffle_LFP = nan(1,numShuffles);
            stability_halves_shuffle_spikeDensity = nan(1,numShuffles);
            stability_interleaved_shuffle_LFP = nan(1,numShuffles);
            stability_interleaved_shuffle_spikeDensity = nan(1,numShuffles);   
            ratTrajCorr_shuffle_LFP = nan(1,numShuffles);
            ratTrajCorr_shuffle_spikeDensity = nan(1,numShuffles);
            ratTrajCorr_immobility_shuffle_LFP = nan(1,numShuffles);
            ratTrajCorr_immobility_shuffle_spikeDensity = nan(1,numShuffles);
            replayTrajCorr_shuffle_LFP = nan(1,numShuffles);
            replayTrajCorr_shuffle_spikeDensity = nan(1,numShuffles);
            ratMVLCorr_shuffle_LFP = nan(1,numShuffles);
            ratMVLCorr_shuffle_spikeDensity = nan(1,numShuffles);
            proxToNearestRewardCorr_shuffle_LFP = nan(1,numShuffles);
            proxToNearestRewardCorr_shuffle_spikeDensity = nan(1,numShuffles);
            distToNearestBarrierCorr_shuffle_LFP = nan(1,numShuffles);
            distToNearestBarrierCorr_shuffle_spikeDensity = nan(1,numShuffles);
            barrierDiffCorr_shuffle_LFP = nan(1,numShuffles);
            barrierDiffCorr_shuffle_spikeDensity = nan(1,numShuffles);
            behaviorDiffCorr_shuffle_LFP = nan(1,numShuffles);
            behaviorDiffCorr_shuffle_spikeDensity = nan(1,numShuffles);
            rippleTimes_shuffle = cell(1,numShuffles);
            for i = 1:numShuffles
                
                randShift = round(rand*size(replays_LFP,1));
                replays_shuffle_LFP = [replays_LFP(:,1) circshift(replays_LFP(:,2:end),randShift)];
                replays_shuffle_spikeDensity = [replays_LFP(:,1) circshift(replays_spikeDensity(:,2:end),randShift)];
                replays_shuffle = replays; replays_shuffle(:,2:end) = circshift(replays_shuffle(:,2:end),randShift);

                randShift = round(rand*size(ripplePower_replayOnly,1));
                ripplePower_replayOnly_shuffle = [ripplePower_replayOnly(:,1) circshift(ripplePower_replayOnly(:,2:end),randShift)];
                    
                %ripple times
                    rippleEvents_shuffle = compute_events_times([ripplePower_replayOnly_shuffle(:,1),ripplePower_replayOnly_shuffle(:,3)],eventZscoreThr,eventMergeThr,eventDurationThr);
                    rippleTimes_shuffle{i} = rippleEvents_shuffle(:,3);

                %compute ripple and PBE fields 
                    [replays_mat_smoothed_shuffle_LFP,replays_mat_smoothed_NaN_shuffle_LFP] = compute_ampField(replays(:,2:3),replays_shuffle_LFP(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
                    [replays_mat_smoothed_shuffle_LFP_nonZscored,replays_mat_smoothed_NaN_shuffle_LFP_nonZscored,~,replays_mat_raw_smoothed_NaN_shuffle_LFP_nonZscored] = compute_ampField(replays(:,2:3),replays_shuffle_LFP(:,end-1),replayDensity_mat,numSpatialBins,gaussFilt);
                    [replays_mat_smoothed_shuffle_spikeDensity,replays_mat_smoothed_NaN_shuffle_spikeDensity] = compute_ampField(replays(:,2:3),replays_shuffle_spikeDensity(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
                    [replays_mat_smoothed_shuffle_spikeDensity_nonZscored,replays_mat_smoothed_NaN_shuffle_spikeDensity_nonZscored,~,replays_mat_raw_smoothed_NaN_shuffle_spikeDensity_nonZscored] = compute_ampField(replays(:,2:3),replays_shuffle_spikeDensity(:,end-1),replayDensity_mat,numSpatialBins,gaussFilt);

                    if i==1, ha = setUp_tightSubplot(7,2); end
                        if i==1
                            axes(ha(2*i-1))
                            imagesc(nan(numSpatialBins))
                            hold on, plot(replays(:,2),replays(:,3),'k','linewidth',3), hold off, plot_mazeProperties
                            hold on, cplot(replays(:,2),replays(:,3),replays_LFP(:,3),'linewidth',1), hold off, 
                            plot_mazeProperties, caxis([0 2.5])
    
                            axes(ha(2*i))
                            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3)) 
                            hold on, imagesc((replays_mat_smoothed_NaN_LFP),'alphadata',~isnan(replays_mat_smoothed_NaN_shuffle_LFP)), hold off,
                            plot_mazeProperties, caxis([0 2.5])
                        else
                            axes(ha(2*i-1))
                            imagesc(nan(numSpatialBins))
                            hold on, plot(replays(:,2),replays(:,3),'k','linewidth',3), hold off, plot_mazeProperties
                            hold on, cplot(replays(:,2),replays(:,3),replays_shuffle_LFP(:,3),'linewidth',1), hold off, 
                            plot_mazeProperties, caxis([0 2.5])
    
                            axes(ha(2*i))
                            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3)) 
                            hold on, imagesc((replays_mat_smoothed_NaN_shuffle_LFP),'alphadata',~isnan(replays_mat_smoothed_NaN_shuffle_LFP)), hold off,
                            plot_mazeProperties, caxis([0 2.5])
                        end
                    drawnow
                    continue
                    if i==7, keyboard, end
                    arrayfun(@(ax) caxis(ax, [0 2]), findall(gcf, 'type', 'axes'));
                    print('shuffles.svg', '-dsvg', '-r800');
                    exportgraphics(gcf, 'shuffles.eps', 'ContentType', 'vector');   


                %spatial info
                    spatialInfo_shuffle_LFP(i) = compute_spatialInfo(replays_mat_smoothed_NaN_shuffle_LFP_nonZscored,replayDensity_mat);
                    spatialInfo_shuffle_spikeDensity(i) = compute_spatialInfo(replays_mat_smoothed_NaN_shuffle_spikeDensity_nonZscored,replayDensity_mat); 

                %SWR vs PBE field
                    LFPvsSpikeDensityCorr_shuffle(i) = compute_corrCoef([replays_mat_smoothed_NaN_LFP(:) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);

                %rate map sum Corr
                    rateMapSumCorr_shuffle_LFP(i) = compute_corrCoef([replays_mat_smoothed_NaN_shuffle_LFP(:) rateMapSum_smoothed_NaN(:)]);
                    rateMapSumCorr_shuffle_spikeDensity(i) = compute_corrCoef([replays_mat_smoothed_NaN_shuffle_spikeDensity(:) rateMapSum_smoothed_NaN(:)]);

                %replay trajectory correlation
                    replayTrajCorr_shuffle_LFP(i) = compute_corrCoef([replayDensity_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
                    replayTrajCorr_shuffle_spikeDensity(i) = compute_corrCoef([replayDensity_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);

                %rat trajectory correlation
                    ratTrajCorr_shuffle_LFP(i) = compute_corrCoef([positionDensity_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
                    ratTrajCorr_shuffle_spikeDensity(i) = compute_corrCoef([positionDensity_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);

                    ratTrajCorr_immobility_shuffle_LFP(i) = compute_corrCoef([positionDensity_immobility_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
                    ratTrajCorr_immobility_shuffle_spikeDensity(i) = compute_corrCoef([positionDensity_immobility_mat_smoothed_NaN(:) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);

                %rat MVL correlation
                    ratMVLCorr_shuffle_LFP(i) = compute_corrCoef([ratMVL(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
                    ratMVLCorr_shuffle_spikeDensity(i) = compute_corrCoef([vectorField_rat{sessionNum}(:,1) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);

                %distance to nearest reward
                    proxToNearestRewardCorr_shuffle_LFP(i) = compute_corrCoef([distToRewardLocs_mat(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
                    proxToNearestRewardCorr_shuffle_spikeDensity(i) = compute_corrCoef([distToRewardLocs_mat(:) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);

                %distance to nearest barrier
                    distToNearestBarrierCorr_shuffle_LFP(i) = compute_corrCoef([distToNearestBarrier_mat(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
                    distToNearestBarrierCorr_shuffle_spikeDensity(i) = compute_corrCoef([distToNearestBarrier_mat(:) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);

                %across-session measures
                    if sessionNum>1
                    %Place field Corr
                        rateMapSumCorr_shuffle_LFP_unstable(i) = compute_corrCoef([replays_mat_smoothed_NaN_shuffle_LFP(:) rateMapSum_unstable_smoothed_NaN(:)]);
                        rateMapSumCorr_shuffle_LFP_stable(i) = compute_corrCoef([replays_mat_smoothed_NaN_shuffle_LFP(:) rateMapSum_stable_smoothed_NaN(:)]);
                        rateMapSumCorr_shuffle_spikeDensity_unstable(i) = compute_corrCoef([replays_mat_smoothed_NaN_shuffle_spikeDensity(:) rateMapSum_unstable_smoothed_NaN(:)]);
                        rateMapSumCorr_shuffle_spikeDensity_stable(i) = compute_corrCoef([replays_mat_smoothed_NaN_shuffle_spikeDensity(:) rateMapSum_stable_smoothed_NaN(:)]);

                    %barrierDiff correlation
                        barrierDiffCorr_shuffle_LFP(i) = compute_corrCoef([barrierDiff(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
                        barrierDiffCorr_shuffle_spikeDensity(i) = compute_corrCoef([barrierDiff(:) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);

                    %Behavioral similarity correlation
                        behaviorDiffCorr_shuffle_LFP(i) = compute_corrCoef([behaviorDiff(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
                        behaviorDiffCorr_shuffle_spikeDensity(i) = compute_corrCoef([behaviorDiff(:) replays_mat_smoothed_NaN_shuffle_spikeDensity(:)]);
                    end

                %stability
                for tt = 1:2
                    spikeDensity_mat_vec = nan(prod(numSpatialBins),2);
                    LFP_mat_vec = nan(prod(numSpatialBins),2);
                    for t = 1:2
                        if tt==1
                            if t==1 %halves
                                edges_sub = timePoints(ind_replay(1:round(length(ind_replay)/2)),:);
                            elseif t==2
                                edges_sub = timePoints(ind_replay(round(length(ind_replay)/2)+1:end),:);
                            end
                        else
                            if t==1 %interleaved
                                edges_sub = timePoints(ind_replay(1:2:end),:);
                            elseif t==2
                                edges_sub = timePoints(ind_replay(2:2:end),:);
                            end
                        end

                        %replays
                        [~,replays_sub] = compute_dataTemporalConcatenation(replays_shuffle,edges_sub);
                        replays_locs_sub = compute_binsToLocs(replays_sub(:,2:3),x_edges,y_edges);
                        replayDensity_mat_sub = hist3([replays_locs_sub(:,2) replays_locs_sub(:,1)],'Edges',{y_edges x_edges}); replayDensity_mat_sub = replayDensity_mat_sub(1:end-1,1:end-1);

                        %SWR
                        replays_LFP_sub = compute_dataInterpolation(replays_LFP_NaNremoved,replays_sub(:,1));
                        [replays_mat_smoothed_LFP_sub,replays_mat_smoothed_NaN_LFP_sub,~,replays_mat_raw_smoothed_NaN_LFP_sub] = compute_ampField(replays_sub(:,2:3),replays_LFP_sub(:,end),replayDensity_mat_sub,numSpatialBins,gaussFilt);
                        LFP_mat_vec(:,t) = replays_mat_smoothed_NaN_LFP_sub(:);

                        %spike density
                        replays_spikeDensity_sub = compute_dataInterpolation(replays_spikeDensity_NaNremoved,replays_sub(:,1));
                        [replays_mat_smoothed_spikeDensity_sub,replays_mat_smoothed_NaN_spikeDensity_sub,~,replays_mat_raw_smoothed_NaN_spikeDensity_sub] = compute_ampField(replays_sub(:,2:3),replays_spikeDensity_sub(:,end),replayDensity_mat_sub,numSpatialBins,gaussFilt);
                        spikeDensity_mat_vec(:,t) = replays_mat_smoothed_NaN_spikeDensity_sub(:);

                    end
                    if tt==1
                        stability_halves_shuffle_spikeDensity(i) = compute_corrCoef(spikeDensity_mat_vec);                    
                        stability_halves_shuffle_LFP(i) = compute_corrCoef(LFP_mat_vec);
                    else
                        stability_interleaved_shuffle_spikeDensity(i) = compute_corrCoef(spikeDensity_mat_vec);                    
                        stability_interleaved_shuffle_LFP(i) = compute_corrCoef(LFP_mat_vec);
                    end 
                end

            end
            %compute pvalues
                LFPvsSpikeDensityCorr_pvalue = (1+nansum(LFPvsSpikeDensityCorr_shuffle>LFPvsSpikeDensityCorr))/(1+numShuffles);
                spatialInfo_pvalue_LFP = (1+nansum(spatialInfo_shuffle_LFP>spatialInfo_LFP))/(1+numShuffles);
                spatialInfo_pvalue_spikeDensity = (1+nansum(spatialInfo_shuffle_spikeDensity>spatialInfo_spikeDensity))/(1+numShuffles);
                stability_halves_pvalue_LFP = (1+nansum(stability_halves_shuffle_LFP>stability_halves_LFP))/(1+numShuffles);
                stability_halves_pvalue_spikeDensity = (1+nansum(stability_halves_shuffle_spikeDensity>stability_halves_spikeDensity))/(1+numShuffles);
                stability_interleaved_pvalue_LFP = (1+nansum(stability_interleaved_shuffle_LFP>stability_interleaved_LFP))/(1+numShuffles);
                stability_interleaved_pvalue_spikeDensity = (1+nansum(stability_interleaved_shuffle_spikeDensity>stability_interleaved_spikeDensity))/(1+numShuffles);
                rateMapCorr_pvalue_LFP = (1+nansum(rateMapSumCorr_shuffle_LFP>rateMapSumCorr_LFP))/(1+numShuffles);
                rateMapCorr_pvalue_spikeDensity = (1+nansum(rateMapSumCorr_shuffle_spikeDensity>rateMapSumCorr_spikeDensity))/(1+numShuffles);
                ratTrajCorr_pvalue_LFP = (1+nansum(ratTrajCorr_shuffle_LFP>ratTrajCorr_LFP))/(1+numShuffles);
                ratTrajCorr_pvalue_spikeDensity = (1+nansum(ratTrajCorr_shuffle_spikeDensity>ratTrajCorr_spikeDensity))/(1+numShuffles);
                ratTrajCorr_immobility_pvalue_LFP = (1+nansum(ratTrajCorr_immobility_shuffle_LFP>ratTrajCorr_immobility_LFP))/(1+numShuffles);
                ratTrajCorr_immobility_pvalue_spikeDensity = (1+nansum(ratTrajCorr_immobility_shuffle_spikeDensity>ratTrajCorr_immobility_spikeDensity))/(1+numShuffles);
                ratMVLCorr_pvalue_LFP = (1+nansum(ratMVLCorr_shuffle_LFP>ratMVLCorr_LFP))/(1+numShuffles);
                ratMVLCorr_pvalue_spikeDensity = (1+nansum(ratMVLCorr_shuffle_spikeDensity>ratMVLCorr_spikeDensity))/(1+numShuffles);
                replayTrajCorr_pvalue_LFP = (1+nansum(replayTrajCorr_shuffle_LFP>replayTrajCorr_LFP))/(1+numShuffles);
                replayTrajCorr_pvalue_spikeDensity = (1+nansum(replayTrajCorr_shuffle_spikeDensity>replayTrajCorr_spikeDensity))/(1+numShuffles);
                proxToNearestRewardCorr_pvalue_LFP = (1+nansum(proxToNearestRewardCorr_shuffle_LFP>proxToNearestRewardCorr_LFP))/(1+numShuffles);
                proxToNearestRewardCorr_pvalue_spikeDensity = (1+nansum(proxToNearestRewardCorr_shuffle_spikeDensity>proxToNearestRewardCorr_spikeDensity))/(1+numShuffles);
                distToNearestBarrierCorr_pvalue_LFP = (1+nansum(distToNearestBarrierCorr_shuffle_LFP>distToNearestBarrierCorr_LFP))/(1+numShuffles);
                distToNearestBarrierCorr_pvalue_spikeDensity = (1+nansum(distToNearestBarrierCorr_shuffle_spikeDensity>distToNearestBarrierCorr_spikeDensity))/(1+numShuffles);

                if sessionNum>1
                    rateMapCorr_pvalue_LFP_unstable = (1+nansum(rateMapSumCorr_shuffle_LFP_unstable>rateMapSumCorr_LFP_unstable))/(1+numShuffles);
                    rateMapCorr_pvalue_LFP_stable = (1+nansum(rateMapSumCorr_shuffle_LFP_stable>rateMapSumCorr_LFP_stable))/(1+numShuffles);
                    rateMapCorr_pvalue_spikeDensity_unstable = (1+nansum(rateMapSumCorr_shuffle_spikeDensity_unstable>rateMapSumCorr_spikeDensity_unstable))/(1+numShuffles);
                    rateMapCorr_pvalue_spikeDensity_stable = (1+nansum(rateMapSumCorr_shuffle_spikeDensity_stable>rateMapSumCorr_spikeDensity_stable))/(1+numShuffles);
                    barrierDiffCorr_pvalue_LFP = (1+nansum(barrierDiffCorr_shuffle_LFP>barrierDiffCorr_LFP))/(1+numShuffles);
                    barrierDiffCorr_pvalue_spikeDensity = (1+nansum(barrierDiffCorr_shuffle_spikeDensity>barrierDiffCorr_spikeDensity))/(1+numShuffles);
                    behaviorDiffCorr_pvalue_LFP = (1+nansum(behaviorDiffCorr_shuffle_LFP>behaviorDiffCorr_LFP))/(1+numShuffles);
                    behaviorDiffCorr_pvalue_spikeDensity = (1+nansum(behaviorDiffCorr_shuffle_spikeDensity>behaviorDiffCorr_spikeDensity))/(1+numShuffles);
                
                    if sum(isnan(barrierDiff(:)))==prod(numSpatialBins), barrierDiffCorr_pvalue_LFP = nan; barrierDiffCorr_pvalue_spikeDensity = nan; end
                else
                    rateMapCorr_pvalue_LFP_unstable = nan;
                    rateMapCorr_pvalue_LFP_stable = nan;
                    rateMapCorr_pvalue_spikeDensity_unstable = nan;
                    rateMapCorr_pvalue_spikeDensity_stable = nan;
                    barrierDiffCorr_pvalue_LFP = nan;
                    barrierDiffCorr_pvalue_spikeDensity = nan;
                    behaviorDiffCorr_pvalue_LFP = nan;
                    behaviorDiffCorr_pvalue_spikeDensity = nan;
                end

%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

    %concatenate
        Data(ii).file = dayFiles{day};
        Data(ii).ratName = Rat_Names{rat};
        Data(ii).sessionNum = sessionNum;
        Data(ii).rat = rat;
        Data(ii).day = day;

        Data(ii).numSpatialBins = numSpatialBins;

        Data(ii).replays = replays;
        Data(ii).ratLocs = ratLoc(ind_replay,:);

        Data(ii).rateMapSum = rateMapSum_smoothed_NaN;
        Data(ii).rateMapSum_unstable = rateMapSum_unstable_smoothed_NaN;
        Data(ii).rateMapSum_stable = rateMapSum_stable_smoothed_NaN;        
        Data(ii).replayDensity_mat = replayDensity_mat;
        Data(ii).replayDensity_smoothed_mat = replayDensity_mat_smoothed_NaN;

        Data(ii).replayDensity_stability_halves = stability_halves_replayDensity;
        Data(ii).replayDensity_stability_interleaved = stability_interleaved_replayDensity;

        Data(ii).positionDensity_mat = positionDensity_mat_smoothed_NaN;
        Data(ii).positionDensity_immobility_mat = positionDensity_immobility_mat_smoothed_NaN;
        Data(ii).positions_mat_MVL = ratMVL;

        Data(ii).distToRewardLocs_mat = reshape(distToRewardLocs_mat,numSpatialBins);
        Data(ii).distToNearestBarrier_mat = reshape(distToNearestBarrier_mat,numSpatialBins);
        Data(ii).behaviorDiff = reshape(behaviorDiff,numSpatialBins);
        Data(ii).barrierDiff = reshape(barrierDiff,numSpatialBins);
        Data(ii).behaviorDiff = reshape(behaviorDiff,numSpatialBins);

        Data(ii).tets_ripplePower = tets_ripplePower;
        Data(ii).tets_numCells = tets_numCells;
        
        Data(ii).LFPvsSpikeDensity.pvalue = LFPvsSpikeDensityCorr_pvalue;
        Data(ii).LFPvsSpikeDensity.statistic = LFPvsSpikeDensityCorr;
        Data(ii).LFPvsSpikeDensity.shuffles = LFPvsSpikeDensityCorr_shuffle;

        %LFP
            Data(ii).LFP.replays_amp = replays_LFP;
            Data(ii).LFP.replays_amp_mat = replays_mat_smoothed_NaN_LFP;
            Data(ii).LFP.replays_amp_mat_tets = replays_mat_smoothed_NaN_LFP_tets;

            Data(ii).LFP.rippleFieldPasses_pvalue = rippleFieldPasses_pvalue;
            Data(ii).LFP.rippleFieldPasses_numPasses = rippleFieldPasses_numPasses;
            Data(ii).LFP.rippleFieldPasses_mvl = rippleFieldPasses_mvl;
            Data(ii).LFP.rippleFieldPasses_mvl_shuffles = rippleFieldPasses_mvl_shuffles;

            Data(ii).LFP.spatialInfo.pvalue = spatialInfo_pvalue_LFP;
            Data(ii).LFP.spatialInfo.statistic = spatialInfo_LFP;
            Data(ii).LFP.spatialInfo.shuffles = spatialInfo_shuffle_LFP;
            Data(ii).LFP.spatialInfo.timeShift = spatialInfo_LFP_timeShift;

            Data(ii).LFP.stability_halves.pvalue = stability_halves_pvalue_LFP;
            Data(ii).LFP.stability_halves.statistic = stability_halves_LFP;
            Data(ii).LFP.stability_halves.shuffles = stability_halves_shuffle_LFP;

            Data(ii).LFP.stability_interleaved.pvalue = stability_interleaved_pvalue_LFP;
            Data(ii).LFP.stability_interleaved.statistic = stability_interleaved_LFP;
            Data(ii).LFP.stability_interleaved.shuffles = stability_interleaved_shuffle_LFP;

            Data(ii).LFP.rateMapCorr.pvalue = rateMapCorr_pvalue_LFP;
            Data(ii).LFP.rateMapCorr.statistic = rateMapSumCorr_LFP;
            Data(ii).LFP.rateMapCorr.shuffles = rateMapSumCorr_shuffle_LFP;

            Data(ii).LFP.rateMapCorr_stable.pvalue = rateMapCorr_pvalue_LFP_stable;
            Data(ii).LFP.rateMapCorr_stable.statistic = rateMapSumCorr_LFP_stable;
            Data(ii).LFP.rateMapCorr_stable.shuffles = rateMapSumCorr_shuffle_LFP_stable;

            Data(ii).LFP.rateMapCorr_unstable.pvalue = rateMapCorr_pvalue_LFP_unstable;
            Data(ii).LFP.rateMapCorr_unstable.statistic = rateMapSumCorr_LFP_unstable;
            Data(ii).LFP.rateMapCorr_unstable.shuffles = rateMapSumCorr_shuffle_LFP_unstable;

            Data(ii).LFP.ratTrajCorr.pvalue = ratTrajCorr_pvalue_LFP;
            Data(ii).LFP.ratTrajCorr.statistic = ratTrajCorr_LFP;
            Data(ii).LFP.ratTrajCorr.shuffles = ratTrajCorr_shuffle_LFP;

            Data(ii).LFP.ratTrajCorr_immobility.pvalue = ratTrajCorr_immobility_pvalue_LFP;
            Data(ii).LFP.ratTrajCorr_immobility.statistic = ratTrajCorr_immobility_LFP;
            Data(ii).LFP.ratTrajCorr_immobility.shuffles = ratTrajCorr_immobility_shuffle_LFP;

            Data(ii).LFP.ratMVLCorr.pvalue = ratMVLCorr_pvalue_LFP;
            Data(ii).LFP.ratMVLCorr.statistic = ratMVLCorr_LFP;
            Data(ii).LFP.ratMVLCorr.shuffles = ratMVLCorr_shuffle_LFP;

            Data(ii).LFP.replayTrajCorr.pvalue = replayTrajCorr_pvalue_LFP;
            Data(ii).LFP.replayTrajCorr.statistic = replayTrajCorr_LFP;
            Data(ii).LFP.replayTrajCorr.shuffles = replayTrajCorr_shuffle_LFP;

            Data(ii).LFP.proxToNearestRewardCorr.pvalue = proxToNearestRewardCorr_pvalue_LFP;
            Data(ii).LFP.proxToNearestRewardCorr.statistic = proxToNearestRewardCorr_LFP;
            Data(ii).LFP.proxToNearestRewardCorr.shuffles = proxToNearestRewardCorr_shuffle_LFP;

            Data(ii).LFP.distToNearestBarrierCorr.pvalue = distToNearestBarrierCorr_pvalue_LFP;
            Data(ii).LFP.distToNearestBarrierCorr.statistic = distToNearestBarrierCorr_LFP;
            Data(ii).LFP.distToNearestBarrierCorr.shuffles = distToNearestBarrierCorr_shuffle_LFP;

            Data(ii).LFP.barrierDiffCorr.pvalue = barrierDiffCorr_pvalue_LFP;
            Data(ii).LFP.barrierDiffCorr.statistic = barrierDiffCorr_LFP;
            Data(ii).LFP.barrierDiffCorr.shuffles = barrierDiffCorr_shuffle_LFP;

            Data(ii).LFP.behaviorDiffCorr.pvalue = behaviorDiffCorr_pvalue_LFP;
            Data(ii).LFP.behaviorDiffCorr.statistic = behaviorDiffCorr_LFP;
            Data(ii).LFP.behaviorDiffCorr.shuffles = behaviorDiffCorr_shuffle_LFP;

        %spike density
            Data(ii).spikeDensity.replays_amp = replays_spikeDensity;
            Data(ii).spikeDensity.replays_amp_mat = replays_mat_smoothed_NaN_spikeDensity;
            Data(ii).spikeDensity.replays_amp_mat_tets = replays_mat_smoothed_NaN_spikeDensity_tets;

            Data(ii).spikeDensity.spatialInfo.pvalue = spatialInfo_pvalue_spikeDensity;
            Data(ii).spikeDensity.spatialInfo.statistic = spatialInfo_spikeDensity;
            Data(ii).spikeDensity.spatialInfo.shuffles = spatialInfo_shuffle_spikeDensity;

            Data(ii).spikeDensity.stability_halves.pvalue = stability_halves_pvalue_spikeDensity;
            Data(ii).spikeDensity.stability_halves.statistic = stability_halves_spikeDensity;
            Data(ii).spikeDensity.stability_halves.shuffles = stability_halves_shuffle_spikeDensity;

            Data(ii).spikeDensity.stability_interleaved.pvalue = stability_interleaved_pvalue_spikeDensity;
            Data(ii).spikeDensity.stability_interleaved.statistic = stability_interleaved_spikeDensity;
            Data(ii).spikeDensity.stability_interleaved.shuffles = stability_interleaved_shuffle_spikeDensity;

            Data(ii).spikeDensity.rateMapCorr.pvalue = rateMapCorr_pvalue_spikeDensity;
            Data(ii).spikeDensity.rateMapCorr.statistic = rateMapSumCorr_spikeDensity;
            Data(ii).spikeDensity.rateMapCorr.shuffles = rateMapSumCorr_shuffle_spikeDensity;

            Data(ii).spikeDensity.rateMapCorr_stable.pvalue = rateMapCorr_pvalue_spikeDensity_stable;
            Data(ii).spikeDensity.rateMapCorr_stable.statistic = rateMapSumCorr_spikeDensity_stable;
            Data(ii).spikeDensity.rateMapCorr_stable.shuffles = rateMapSumCorr_shuffle_spikeDensity_stable;

            Data(ii).spikeDensity.rateMapCorr_unstable.pvalue = rateMapCorr_pvalue_spikeDensity_unstable;
            Data(ii).spikeDensity.rateMapCorr_unstable.statistic = rateMapSumCorr_spikeDensity_unstable;
            Data(ii).spikeDensity.rateMapCorr_unstable.shuffles = rateMapSumCorr_shuffle_spikeDensity_unstable;

            Data(ii).spikeDensity.ratTrajCorr.pvalue = ratTrajCorr_pvalue_spikeDensity;
            Data(ii).spikeDensity.ratTrajCorr.statistic = ratTrajCorr_spikeDensity;
            Data(ii).spikeDensity.ratTrajCorr.shuffles = ratTrajCorr_shuffle_spikeDensity;

            Data(ii).spikeDensity.ratTrajCorr_immobility.pvalue = ratTrajCorr_immobility_pvalue_spikeDensity;
            Data(ii).spikeDensity.ratTrajCorr_immobility.statistic = ratTrajCorr_immobility_spikeDensity;
            Data(ii).spikeDensity.ratTrajCorr_immobility.shuffles = ratTrajCorr_immobility_shuffle_spikeDensity;

            Data(ii).spikeDensity.ratMVLCorr.pvalue = ratMVLCorr_pvalue_spikeDensity;
            Data(ii).spikeDensity.ratMVLCorr.statistic = ratMVLCorr_spikeDensity;
            Data(ii).spikeDensity.ratMVLCorr.shuffles = ratMVLCorr_shuffle_spikeDensity;

            Data(ii).spikeDensity.replayTrajCorr.pvalue = replayTrajCorr_pvalue_spikeDensity;
            Data(ii).spikeDensity.replayTrajCorr.statistic = replayTrajCorr_spikeDensity;
            Data(ii).spikeDensity.replayTrajCorr.shuffles = replayTrajCorr_shuffle_spikeDensity;

            Data(ii).spikeDensity.proxToNearestRewardCorr.pvalue = proxToNearestRewardCorr_pvalue_spikeDensity;
            Data(ii).spikeDensity.proxToNearestRewardCorr.statistic = proxToNearestRewardCorr_spikeDensity;
            Data(ii).spikeDensity.proxToNearestRewardCorr.shuffles = proxToNearestRewardCorr_shuffle_spikeDensity;

            Data(ii).spikeDensity.distToNearestBarrierCorr.pvalue = distToNearestBarrierCorr_pvalue_spikeDensity;
            Data(ii).spikeDensity.distToNearestBarrierCorr.statistic = distToNearestBarrierCorr_spikeDensity;
            Data(ii).spikeDensity.distToNearestBarrierCorr.shuffles = distToNearestBarrierCorr_shuffle_spikeDensity;

            Data(ii).spikeDensity.barrierDiffCorr.pvalue = barrierDiffCorr_pvalue_spikeDensity;
            Data(ii).spikeDensity.barrierDiffCorr.statistic = barrierDiffCorr_spikeDensity;
            Data(ii).spikeDensity.barrierDiffCorr.shuffles = barrierDiffCorr_shuffle_spikeDensity;

            Data(ii).spikeDensity.behaviorDiffCorr.pvalue = behaviorDiffCorr_pvalue_spikeDensity;
            Data(ii).spikeDensity.behaviorDiffCorr.statistic = behaviorDiffCorr_spikeDensity;
            Data(ii).spikeDensity.behaviorDiffCorr.shuffles = behaviorDiffCorr_shuffle_spikeDensity;

    %%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%

        ii = ii+1;
    end

    cd ..
end
cd ../..
cd analysis_RippleField/
save('replayEvents_rippleFieldStats_allCells_sigReplays.mat','Data','-v7.3')
cd ..
end
keyboard


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd analysis_RippleField/
% load replayEvents_rippleFieldStats
% load replayEvents_rippleFieldStats_allCells

load replayEvents_rippleFieldStats_allCells_sigReplays

% load replayEvents_SWRfield_6
% load replayEvents_SWRfield_7

% load XXXreplayEvents_SWRfield_8_rateMapCorrected_2
% load replayEvents_SWRfield_lowDispersionReplay

% load replayEvents_rippleFieldStats_short_placeCells
% load replayEvents_rippleFieldStats_short_allCells

cd ..

    Data_LFP = [Data.LFP];
    Data_spikeDensity = [Data.spikeDensity];
    Data_LFPvsSpikeDensity = [Data.LFPvsSpikeDensity];
    
    Data_LFP_spatialInfo = [Data_LFP.spatialInfo]; 
    Data_LFP_stability_interleaved = [Data_LFP.stability_interleaved]; 
    Data_LFP_stability_halves = [Data_LFP.stability_halves]; 
    Data_LFP_rateMapCorr = [Data_LFP.rateMapCorr]; 
    Data_LFP_replayTrajCorr = [Data_LFP.replayTrajCorr]; 
    Data_LFP_ratTrajCorr = [Data_LFP.ratTrajCorr]; 
    Data_LFP_ratTrajCorr_immobility = [Data_LFP.ratTrajCorr_immobility]; 
    Data_LFP_rateMapCorr_stable = [Data_LFP.rateMapCorr_stable]; 
    Data_LFP_rateMapCorr_unstable = [Data_LFP.rateMapCorr_unstable]; 
    Data_LFP_ratMVLCorr = [Data_LFP.ratMVLCorr]; 
    Data_LFP_barrierDiffCorr = [Data_LFP.barrierDiffCorr]; 
    Data_LFP_behaviorDiffCorr = [Data_LFP.behaviorDiffCorr]; 
    Data_LFP_distToBarrierCorr = [Data_LFP.distToNearestBarrierCorr]; 
    Data_LFP_distToRewardCorr = [Data_LFP.proxToNearestRewardCorr]; 
   
    Data_spikeDensity_spatialInfo = [Data_spikeDensity.spatialInfo]; 
    Data_spikeDensity_stability_interleaved = [Data_spikeDensity.stability_interleaved]; 
    Data_spikeDensity_stability_halves = [Data_spikeDensity.stability_halves]; 
    Data_spikeDensity_rateMapCorr = [Data_spikeDensity.rateMapCorr]; 
    Data_spikeDensity_replayTrajCorr = [Data_spikeDensity.replayTrajCorr]; 
    Data_spikeDensity_ratTrajCorr = [Data_spikeDensity.ratTrajCorr]; 
    Data_spikeDensity_ratTrajCorr_immobility = [Data_spikeDensity.ratTrajCorr_immobility]; 
    Data_spikeDensity_rateMapCorr_stable = [Data_spikeDensity.rateMapCorr_stable]; 
    Data_spikeDensity_rateMapCorr_unstable = [Data_spikeDensity.rateMapCorr_unstable]; 
    Data_spikeDensity_ratMVLCorr = [Data_spikeDensity.ratMVLCorr]; 
    Data_spikeDensity_barrierDiffCorr = [Data_spikeDensity.barrierDiffCorr]; 
    Data_spikeDensity_behaviorDiffCorr = [Data_spikeDensity.behaviorDiffCorr]; 
    Data_spikeDensity_distToBarrierCorr = [Data_spikeDensity.distToNearestBarrierCorr]; 
    Data_spikeDensity_distToRewardCorr = [Data_spikeDensity.proxToNearestRewardCorr]; 

    Data_LFP_rippleFieldPasses_pvalue = [];
    Data_LFP_rippleFieldPasses_numPasses = [];
    Data_LFP_rippleFieldPasses_mvl = [];
    Data_LFP_rippleFieldPasses_mvl_shuffles = [];
    for i = 1:length(Data)
        Data_LFP_rippleFieldPasses_pvalue = [Data_LFP_rippleFieldPasses_pvalue; Data_LFP(i).rippleFieldPasses_pvalue];
        Data_LFP_rippleFieldPasses_numPasses = [Data_LFP_rippleFieldPasses_numPasses; Data_LFP(i).rippleFieldPasses_numPasses];
        Data_LFP_rippleFieldPasses_mvl = [Data_LFP_rippleFieldPasses_mvl; Data_LFP(i).rippleFieldPasses_mvl];
        Data_LFP_rippleFieldPasses_mvl_shuffles = [Data_LFP_rippleFieldPasses_mvl_shuffles; Data_LFP(i).rippleFieldPasses_mvl_shuffles];
    end

    pvalues_LFP = [...
    [Data_LFP_spatialInfo.pvalue];... 
    [Data_LFP_stability_halves.pvalue];... 
    [Data_LFP_replayTrajCorr.pvalue];... 
    [Data_LFP_ratTrajCorr.pvalue];...
    [Data_LFP_ratTrajCorr_immobility.pvalue];... 
    [Data_LFP_ratMVLCorr.pvalue];...
    [Data_LFP_distToBarrierCorr.pvalue];...
    [Data_LFP_distToRewardCorr.pvalue];...
    [Data_LFP_barrierDiffCorr.pvalue];...
    [Data_LFP_behaviorDiffCorr.pvalue];...
    [Data_LFPvsSpikeDensity.pvalue];... 
    [Data_LFP_rateMapCorr.pvalue]];

    pvalues_spikeDensity = [...
    [Data_spikeDensity_spatialInfo.pvalue];... 
    [Data_spikeDensity_stability_halves.pvalue];... 
    [Data_spikeDensity_replayTrajCorr.pvalue];... 
    [Data_spikeDensity_ratTrajCorr.pvalue];... 
    [Data_spikeDensity_ratTrajCorr_immobility.pvalue];... 
    [Data_spikeDensity_ratMVLCorr.pvalue];...
    [Data_spikeDensity_distToBarrierCorr.pvalue];...
    [Data_spikeDensity_distToRewardCorr.pvalue];...
    [Data_spikeDensity_barrierDiffCorr.pvalue];...
    [Data_spikeDensity_behaviorDiffCorr.pvalue];...
    [Data_LFPvsSpikeDensity.pvalue];... 
    [Data_spikeDensity_rateMapCorr.pvalue]];

    statistic_LFP = [...
    [Data_LFP_spatialInfo.statistic];... 
    [Data_LFP_stability_halves.statistic];... 
    [Data_LFP_replayTrajCorr.statistic];... 
    [Data_LFP_ratTrajCorr.statistic];... 
    [Data_LFP_ratTrajCorr_immobility.statistic];...  
    [Data_LFP_ratMVLCorr.statistic];... 
    [Data_LFP_distToBarrierCorr.statistic];...
    [Data_LFP_distToRewardCorr.statistic];...
    [Data_LFP_barrierDiffCorr.statistic];...
    [Data_LFP_behaviorDiffCorr.statistic];...
    [Data_LFPvsSpikeDensity.statistic];... 
    [Data_LFP_rateMapCorr.statistic]];

    statistic_spikeDensity = [...
    [Data_spikeDensity_spatialInfo.statistic];... 
    [Data_spikeDensity_stability_halves.statistic];...
    [Data_spikeDensity_replayTrajCorr.statistic];...
    [Data_spikeDensity_ratTrajCorr.statistic];...  
    [Data_spikeDensity_ratTrajCorr_immobility.statistic];...  
    [Data_spikeDensity_ratMVLCorr.statistic];... 
    [Data_spikeDensity_distToBarrierCorr.statistic];...
    [Data_spikeDensity_distToRewardCorr.statistic];...
    [Data_spikeDensity_barrierDiffCorr.statistic];...
    [Data_spikeDensity_behaviorDiffCorr.statistic];... 
    [Data_LFPvsSpikeDensity.statistic];... 
    [Data_spikeDensity_rateMapCorr.statistic]];

    numShuffles = length(Data(1).LFP.spatialInfo.shuffles);
    shuffle_LFP = [...
    [reshape(load_fieldVec(Data_LFP_spatialInfo,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_LFP_stability_halves,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_LFP_replayTrajCorr,'shuffles',numShuffles)',[],1)'];...
    [reshape(load_fieldVec(Data_LFP_ratTrajCorr,'shuffles',numShuffles)',[],1)'];...
    [reshape(load_fieldVec(Data_LFP_ratTrajCorr_immobility,'shuffles',numShuffles)',[],1)'];...  
    [reshape(load_fieldVec(Data_LFP_ratMVLCorr,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_LFP_distToBarrierCorr,'shuffles',numShuffles)',[],1)'];...
    [reshape(load_fieldVec(Data_LFP_distToRewardCorr,'shuffles',numShuffles)',[],1)'];...
    [reshape(load_fieldVec(Data_LFP_barrierDiffCorr,'shuffles',numShuffles)',[],1)'];...
    [reshape(load_fieldVec(Data_LFP_behaviorDiffCorr,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_LFPvsSpikeDensity,'shuffles',numShuffles)',[],1)'];...
    [reshape(load_fieldVec(Data_LFP_rateMapCorr,'shuffles',numShuffles)',[],1)']]; 
  
    shuffle_spikeDensity = [...
    [reshape(load_fieldVec(Data_spikeDensity_spatialInfo,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_spikeDensity_stability_halves,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_spikeDensity_replayTrajCorr,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_spikeDensity_ratTrajCorr,'shuffles',numShuffles)',[],1)'];...   
    [reshape(load_fieldVec(Data_spikeDensity_ratTrajCorr_immobility,'shuffles',numShuffles)',[],1)'];...    
    [reshape(load_fieldVec(Data_spikeDensity_ratMVLCorr,'shuffles',numShuffles)',[],1)'];...   
    [reshape(load_fieldVec(Data_spikeDensity_distToBarrierCorr,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_spikeDensity_distToRewardCorr,'shuffles',numShuffles)',[],1)'];...   
    [reshape(load_fieldVec(Data_spikeDensity_barrierDiffCorr,'shuffles',numShuffles)',[],1)'];...
    [reshape(load_fieldVec(Data_spikeDensity_behaviorDiffCorr,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_LFPvsSpikeDensity,'shuffles',numShuffles)',[],1)'];... 
    [reshape(load_fieldVec(Data_spikeDensity_rateMapCorr,'shuffles',numShuffles)',[],1)']]; 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot mosaic: replays, replays w/ ripples, ripple field, stability, PFs, PBE field
     
    % ind_select = [2,4,6,7];
    % ind_select = [5,6,21,33];
    ind_select = [5,9,21,33];
    Data_LFP = [Data.LFP];
    Data_spikeDensity = [Data.spikeDensity];
    for ii = 1:length(Data)
        if ~ismember(ii,ind_select), continue, end
        
        figure
        cd(strcat(Data(ii).ratName,'/processed/',Data(ii).file))
        load Experiment_Information
        load Analysis_Information
        load behavior
        load clusters
        sessionNum = Data(ii).sessionNum; sessionNum_decoder = sessionNum;
        cd ../../..
               
            % replays = Data(ii).replays;
            % replays_spikeDensity = Data(ii).spikeDensity.replays_amp;
            % replays_LFP = Data(ii).LFP.replays_amp;
            % 
            % replays_locs = compute_binsToLocs(replays(:,2:3),x_edges,y_edges);
            % replayDensity_mat = hist3([replays_locs(:,2) replays_locs(:,1)],'Edges',{y_edges x_edges}); replayDensity_mat = replayDensity_mat(1:end-1,1:end-1);
            % 
            % numSpatialBins_smoothing = 8;
            % gaussFilt = setUp_gaussFilt(numSpatialBins,numSpatialBins_smoothing/binSize);
            % 
            % [replays_mat_smoothed_LFP,replays_mat_smoothed_NaN_LFP] = compute_ampField(replays(:,2:3),replays_LFP(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
            % [replays_mat_smoothed_spikeDensity,replays_mat_smoothed_NaN_spikeDensity] = compute_ampField(replays(:,2:3),replays_spikeDensity(:,end),replayDensity_mat,numSpatialBins,gaussFilt);

            % Data_LFP(ii).replays_amp_mat = replays_mat_smoothed_NaN_LFP;
            % Data_spikeDensity(ii).replays_amp_mat = replays_mat_smoothed_NaN_spikeDensity;
                
            % subplot(121), imagesc(replays_mat_smoothed_NaN_LFP), plot_mazeProperties
            % subplot(122), imagesc(Data(ii).LFP.replays_amp_mat), plot_mazeProperties
            % keyboard
            % continue

            % A = Data(ii).rateMapSum;
            % B = Data_LFP(ii).replays_amp_mat;
            % C = Data_spikeDensity(ii).replays_amp_mat;
            % subplot(231), imagesc(A), plot_mazeProperties, title('rate map sum','color','k')
            % subplot(232), imagesc(B), plot_mazeProperties, title('ripple field','color','r')
            % subplot(233), imagesc(C), plot_mazeProperties, title('spike density field','color','b')
            % subplot(614), plot(compute_zscore(A(:)),'linewidth',2,'color','k'), hold on, plot(compute_zscore(B(:)),'linewidth',2,'color','r'), hold off
            %     title(Data_LFP(ii).rateMapCorr.statistic)
            % subplot(615), plot(compute_zscore(A(:)),'linewidth',2,'color','k'), hold on, plot(compute_zscore(C(:)),'linewidth',2,'color','b'), hold off
            %     title(Data_spikeDensity(ii).rateMapCorr.statistic)
            % subplot(616), plot(compute_zscore(B(:)),'linewidth',2,'color','r'), hold on, plot(compute_zscore(C(:)),'linewidth',2,'color','b'), hold off
            %     title(Data(ii).LFPvsSpikeDensity.statistic)
            % keyboard, continue

        set(gcf,'color','w')
        subplot(161)
        plot(Data(ii).replays(:,2),Data(ii).replays(:,3),'color',0.3*ones(1,3),'linewidth',1);
        hold on, plot(Data(ii).ratLocs(:,1),Data(ii).ratLocs(:,2),'r.','markersize',12); hold off
        set(gca,'ydir','normal'), colorStyle = 'k'; plot_mazeProperties
        title('replays','fontsize',10)
        
        subplot(162)
        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        hold on, plot(Data(ii).replays(:,2),Data(ii).replays(:,3),'k','linewidth',3), hold off
        hold on, cplot(Data(ii).replays(:,2),Data(ii).replays(:,3),Data_LFP(ii).replays_amp(:,end),'linewidth',1), hold off
        set(gca,'ydir','normal'), colorStyle = 'k'; plot_mazeProperties, caxis([0 2])
        title('replays w/ ripples','fontsize',10) 

        subplot(163)
        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        % hold on, imagesc(compute_zscore_squareMatrix(Data_LFP(ii).replays_amp_mat),'alphadata',~isnan(Data_LFP(ii).replays_amp_mat)), hold off
        hold on, imagesc((Data_LFP(ii).replays_amp_mat),'alphadata',~isnan(Data_LFP(ii).replays_amp_mat)), hold off
        set(gca,'fontsize',10,'ydir','normal'), plot_mazeProperties, caxis([0 2])
        title('ripple field','fontsize',10)

        for iii=1:2
            subplot(2,12,12*(iii-1)+7)
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc((Data(ii).LFP.stability_halves.window(iii).replays_amp_mat),'alphadata',~isnan(Data_LFP(ii).stability_interleaved.window(iii).replays_amp_mat)), hold off
            plot_mazeProperties, set(gca,'ydir','normal'), caxis([0 2]), axis square
            if iii==1, title('1st half','fontsize',10), else, title('2nd half','fontsize',10), end
        end
                
        subplot(3,12,8)
        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        hold on, imagesc(compute_zscore_squareMatrix(Data(ii).replayDensity_smoothed_mat),'alphadata',~isnan(Data(ii).replayDensity_mat)), hold off
        set(gca,'ydir','normal'), plot_mazeProperties, caxis([0 2])
        title('replay density','fontsize',10)
        
        subplot(3,12,20)
        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        hold on, imagesc(compute_zscore_squareMatrix(Data(ii).positionDensity_immobility_mat),'alphadata',~isnan(Data(ii).positionDensity_mat)), hold off
        set(gca,'ydir','normal'), plot_mazeProperties, caxis([0 2])
        title('rat @ replay','fontsize',10)

        subplot(3,12,32)
        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        hold on, imagesc(compute_zscore_squareMatrix(Data(ii).positionDensity_mat),'alphadata',~isnan(Data(ii).positionDensity_mat)), hold off
        set(gca,'ydir','normal'), plot_mazeProperties, caxis([0 2])
        title('rat movement','fontsize',10)
        
        subplot(165)
        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3)) 
        hold on, imagesc(compute_zscore_squareMatrix(Data(ii).rateMapSum),'alphadata',~isnan(Data(ii).rateMapSum)), hold off
        set(gca,'ydir','normal'), plot_mazeProperties, caxis([-1 2])
        title('place field sum','fontsize',10)
        
            %plot place field centers            
            [ind_cluster,~,~,~,~,~,rateMap_smoothed_NaN_all,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,ind_cluster_all] = load_rateMapsForDecoding_2D(clusters,sessionNum_decoder,Run_Times,numSpatialBins,meanFiringRateThr,spatialInfoThr,noiseOverlapThr,isolationThr,peakSnrThr,numSpatialBins_coarse);
            if use_placeCells==0,ind_cluster = ind_cluster_all; rateMap_smoothed_NaN = rateMap_smoothed_NaN_all; end
            clusters_runs = [clusters.runs]; clusters_runs = clusters_runs(sessionNum_decoder:size(Run_Times,1):end);
            for i = 1:length(ind_cluster)
                maxSpatialFiringBinLoc = compute_maxSpatialFiringLoc(reshape(rateMap_smoothed_NaN(i,:),numSpatialBins),spatialDim);
                hold on, plot(maxSpatialFiringBinLoc(1),maxSpatialFiringBinLoc(2),'r.','markersize',12), hold off
            end
        
        subplot(166)
        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        hold on, imagesc((Data_spikeDensity(ii).replays_amp_mat),'alphadata',~isnan(Data_spikeDensity(ii).replays_amp_mat)), hold off
        set(gca,'ydir','normal'), plot_mazeProperties, caxis([0 2]) 
        title('spike density field','fontsize',10) 

        % keyboard
        drawnow
    end


%plot mosaic: ripple field, prox to reward, dist to barriers, rat MVL, barrier diff, behavior diff

    ind_select = [5,9,21,33];
    Data_LFP = [Data.LFP];
    Data_spikeDensity = [Data.spikeDensity];
    for ii = 1:length(Data)
        if ~ismember(ii,ind_select), continue, end
        
        figure
        cd(strcat(Data(ii).ratName,'/processed/',Data(ii).file))
        load Experiment_Information
        load Analysis_Information
        sessionNum = Data(ii).sessionNum; sessionNum_decoder = sessionNum;
        cd ../../..

        subplot(161)
        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        hold on, imagesc((Data_LFP(ii).replays_amp_mat),'alphadata',~isnan(Data_LFP(ii).replays_amp_mat)), hold off
        set(gca,'fontsize',10,'ydir','normal'), plot_mazeProperties, caxis([0 1.5])
        title('ripple field','fontsize',10)

        subplot(162)
        imagesc(compute_zscore_squareMatrix(Data(ii).positionDensity_mat))
        set(gca,'fontsize',10,'ydir','normal'), plot_mazeProperties, 
        title('prox. to reward','fontsize',10)
        xlabel(strcat('r=',num2str(compute_round(Data(ii).LFP.ratTrajCorr.statistic,100)),load_pstar(Data(ii).LFP.ratTrajCorr.pvalue)))
        caxis([-1 2.5])

        subplot(163)
        imagesc(Data(ii).distToRewardLocs_mat)
        set(gca,'fontsize',10,'ydir','normal'), plot_mazeProperties, 
        title('prox. to reward','fontsize',10)
        xlabel(strcat('r=',num2str(compute_round(Data(ii).LFP.proxToNearestRewardCorr.statistic,100)),load_pstar(Data(ii).LFP.proxToNearestRewardCorr.pvalue)))

        subplot(164)
        imagesc(Data(ii).distToNearestBarrier_mat)
        set(gca,'fontsize',10,'ydir','normal'), plot_mazeProperties, 
        title('dist. to barrier','fontsize',10)
        xlabel(strcat('r=',num2str(compute_round(Data(ii).LFP.distToNearestBarrierCorr.statistic,100)),load_pstar(Data(ii).LFP.distToNearestBarrierCorr.pvalue)))

        subplot(165)
        imagesc(Data(ii).barrierDiff)
        set(gca,'fontsize',10,'ydir','normal'), plot_mazeProperties, 
        title('barrier diff.','fontsize',10)
        xlabel(strcat('r=',num2str(compute_round(Data(ii).LFP.barrierDiffCorr.statistic,100)),load_pstar(Data(ii).LFP.barrierDiffCorr.pvalue)))

        subplot(166)
        imagesc(Data(ii).behaviorDiff)
        set(gca,'fontsize',10,'ydir','normal'), plot_mazeProperties,
        title('behavior diff.','fontsize',10)
        xlabel(strcat('r=',num2str(compute_round(Data(ii).LFP.behaviorDiffCorr.statistic,100)),load_pstar(Data(ii).LFP.behaviorDiffCorr.pvalue)))

        % keyboard
        drawnow
    end


        % subplot(121)
        % imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        % hold on, imagesc(compute_zscore_squareMatrix(Data_LFP(ii).replays_amp_mat),'alphadata',~isnan(Data_LFP(ii).replays_amp_mat)), hold off
        % set(gca,'fontsize',10,'ydir','normal'), plot_mazeProperties, caxis([-1 2.5])
        % title('ripple field','fontsize',10)
        % 
        % subplot(122)
        % imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        % hold on, imagesc(compute_zscore_squareMatrix(Data_spikeDensity(ii).replays_amp_mat),'alphadata',~isnan(Data_spikeDensity(ii).replays_amp_mat)), hold off
        % set(gca,'ydir','normal'), plot_mazeProperties, caxis([-1 2.5]) 
        % title('spike density field','fontsize',10) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot pvalue and statistic distributions

    names = {'spatial info.','stability','replay density corr.',{'rat density corr.','(movement)'},{'rat density corr.','(immobility)'},...
            'rat MVL corr.','barrier distance corr.','reward proximity corr.',{'barrier dissimilarity corr.'},{'behavior dissimilarity corr.'},...
            'ripple vs. spike density corr.','place field sum corr.'};

    numBins = 15;
    edges = {linspace(0,0.05,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),...
             linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),...
             linspace(-0.5,1,numBins),linspace(-0.5,1,numBins)};
    
        colors = {'k','b'};
        statistic = statistic_LFP; 
        pvalues = pvalues_LFP; 
        shuffle = shuffle_LFP;
        for i = 1:length(names)
            [X2, p_combined] = fishers_combined_method(pvalues(i,~isnan(pvalues(i,:))));
    
            fprintf('x = %.15f\n', p_combined)

            figure(1), set(gcf,'color','w')
            subplot(3,5,i)
            histogram(statistic(i,:),edges{i},'facecolor',colors{1},'edgecolor','none','normalization','probability')
            h = histcounts(shuffle(i,:),edges{i},'normalization','probability'); hold on, plot(edges{i}(1:end-1)+mean(diff(edges{i}))/2,h,colors{1},'linewidth',2), hold off
            axis square, xlabel(names{i}), ylabel('fraction of sessions'), set(gca,'fontsize',12), axis tight, box on
            if i==1, legend('data','shuffle'), end
            xl = xlim; yl = ylim;
            % text(0.05+diff(xl)*0.2,yl(2)-diff(yl)*0.1,strcat(num2str(nansum(pvalues(i,:)<=0.05)),'/',num2str(sum(~isnan(pvalues(i,:))))),'verticalAlignment','top','horizontalAlignment','left','fontsize',12,'color',colors{1})
    
            if i==12 %place field sum
                hold on, histogram(statistic_spikeDensity(i,:),edges{i},'facecolor',colors{2},'edgecolor','none','normalization','probability'), hold off, 
                h = histcounts(shuffle_spikeDensity(i,:),edges{i},'normalization','probability'); hold on, plot(edges{i}(1:end-1)+mean(diff(edges{i}))/2,h,colors{2},'linewidth',2), hold off
            end
    
            figure(2), set(gcf,'color','w')
            subplot(3,5,i)
            histogram(pvalues(i,:),linspace(0,1,20),'normalization','probability','facecolor',colors{1},'edgecolor','none')
            axis square, xlabel('p-value'), ylabel('fraction of sessions'), set(gca,'fontsize',12), axis tight, vline(0.05,'k--'), box on
            title(names{i},'fontweight','normal','fontsize',12)
            xl = xlim; yl = ylim; 
            text(0.05+diff(xl)*0.2,yl(2)-diff(yl)*0.1,strcat(num2str(nansum(pvalues(i,:)<=0.05)),'/',num2str(sum(~isnan(pvalues(i,:)))),load_pstar(p_combined)),'verticalAlignment','top','horizontalAlignment','left','fontsize',12,'color',colors{1})
    
            if i==12 %place field sum
                hold on, histogram(pvalues_spikeDensity(i,:),linspace(0,1,20),'normalization','probability','facecolor',colors{2},'edgecolor','none'), hold off,
                text(0.05+diff(xl)*0.2,yl(2)-diff(yl)*0.2,strcat(num2str(nansum(pvalues_spikeDensity(i,:)<=0.05)),'/',num2str(sum(~isnan(pvalues_spikeDensity(i,:)))),load_pstar(p_combined)),'verticalAlignment','top','horizontalAlignment','left','fontsize',12,'color',colors{2})
                vline(0.05,'k--')
            end
        end

    %directionality
        figure(1)
        subplot(3,5,13)
        edges_mvl = linspace(0.5,1,numBins); centers_mvl = edges_mvl(1:end-1) + mean(diff(edges_mvl))/2;
        histogram(Data_LFP_rippleFieldPasses_mvl(Data_LFP_rippleFieldPasses_numPasses>60),edges_mvl,'normalization','probability','edgecolor','none','facecolor','k')
        h = histcounts(reshape(Data_LFP_rippleFieldPasses_mvl_shuffles(Data_LFP_rippleFieldPasses_numPasses>40,:),1,[]),edges_mvl,'normalization','probability');
        hold on, plot(centers_mvl,h,'k','linewidth',2), hold off
        axis square, xlabel('directionality'), ylabel('fraction of fields'), set(gca,'fontsize',12), set(gcf,'color','w'), axis tight, box on
        xl = xlim; yl = ylim; 
        text(xl(1)+diff(xl)*0.2,yl(2)-diff(yl)*0.1,strcat(num2str(sum(Data_LFP_rippleFieldPasses_pvalue(Data_LFP_rippleFieldPasses_numPasses>30)<=0.05)),'/',num2str(sum(~isnan(Data_LFP_rippleFieldPasses_pvalue(Data_LFP_rippleFieldPasses_numPasses>30))))),'verticalAlignment','top','horizontalAlignment','left','fontsize',12,'color','k')
    
        figure(2)
        subplot(3,5,13)
        edges_pvalue = linspace(0,1,20);
        [X2, p_combined] = fishers_combined_method(Data_LFP_rippleFieldPasses_pvalue(Data_LFP_rippleFieldPasses_numPasses>30));
        fprintf('x = %.15f', p_combined)
        histogram(Data_LFP_rippleFieldPasses_pvalue(Data_LFP_rippleFieldPasses_numPasses>30),edges_pvalue,'normalization','probability','edgecolor','none','facecolor','k')
        axis square, xlabel('p-value'), ylabel('fraction of fields'), set(gca,'fontsize',12), set(gcf,'color','w'), axis tight, vline(0.05,'k--'), box on
        title('directionality','fontweight','normal','fontsize',12)
        xl = xlim; yl = ylim; 
        text(xl(1)+diff(xl)*0.2,yl(2)-diff(yl)*0.1,strcat(num2str(sum(Data_LFP_rippleFieldPasses_pvalue(Data_LFP_rippleFieldPasses_numPasses>30)<=0.05)),'/',num2str(sum(~isnan(Data_LFP_rippleFieldPasses_pvalue(Data_LFP_rippleFieldPasses_numPasses>30)))),load_pstar(p_combined)),'verticalAlignment','top','horizontalAlignment','left','fontsize',12,'color','k')

    %stable vs unstable cells
        correlations_LFP = [[Data_LFP_rateMapCorr_stable.statistic]; [Data_LFP_rateMapCorr_unstable.statistic]]';
        correlations_spikeDensity = [[Data_spikeDensity_rateMapCorr_stable.statistic];[Data_spikeDensity_rateMapCorr_unstable.statistic]]';
        figure(1)
        for i = 1:2
            subplot(3,5,13+i)
            if i==1, correlations = correlations_LFP; else, correlations = correlations_spikeDensity; end
            histogram(correlations(:,1),linspace(-0.5,1,numBins),'facecolor','g','edgecolor','none'), hold on, histogram(correlations(:,2),linspace(-1,1,numBins),'facecolor','m','edgecolor','none')
            vline(nanmean(correlations(:,1)),'k'), vline(nanmean(correlations(:,2)),'k')
            vline(nanmean(correlations(:,1)),'g--'), vline(nanmean(correlations(:,2)),'m--')
            xlabel('correlation'), axis square, ylabel('count'), set(gca,'fontsize',12), axis tight, box on
            set(gcf,'color','w'), if i==1, legend('stable','unstable'), end
            % if i==1, title('ripple','fontweight','normal'), else, title('spike density','fontweight','normal'), end
            [~,p] = ttest(correlations(:,1),correlations(:,2)); 
            xl = xlim; yl = ylim; 
            xlim([-0.6 1])
            text(xl(1),yl(2),load_pstar(p),'verticalAlignment','top','horizontalAlignment','left','fontsize',10,'color','k')
        end
        

    %%%%%%%%%%%%%%%%

    %spatial info vs ripple time within replay (time shifted ripple power forwards and backwards)
        spatialInfo_LFP_timeShift = [];
        for i = 1:37
            spatialInfo_LFP_timeShift = [spatialInfo_LFP_timeShift; Data_LFP_spatialInfo(i).timeShift];
        end
        spatialInfo_LFP_timeShift(15,:) = nan;
            
        shifts = -0.5:0.05:0.5;    
        shadedErrorBar(shifts,nanmean(spatialInfo_LFP_timeShift),nanstd(spatialInfo_LFP_timeShift)./sqrt(size(spatialInfo_LFP_timeShift,1)),'lineprops',{'k','linewidth',2})
            xlabel('time shift (sec)'), ylabel('spatial info.'), set(gca,'fontsize',14), axis square, set(gcf,'color','w'), box on

    %%%%%%%%%%%%%%%%

    %individual examples
        names = {'spatial info.','stability','replay density corr.',{'rat density corr.','(movement)'},{'rat density corr.','(immobility)'},'directionality'};
        numBins = 20;
        edges = {linspace(0,0.06,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins),linspace(0.5,1,20)};
        ind_select = [5,6,21,33];
        colors = {'r','b','g','m'};
        for i = 1:6
            for j = 1:4
                subplot(4,6,i+(j-1)*6)
                if i==1, statistic = Data(ind_select(j)).LFP.spatialInfo.statistic; shuffles = Data(ind_select(j)).LFP.spatialInfo.shuffles; pvalue = Data(ind_select(j)).LFP.spatialInfo.pvalue;
                elseif i==2, statistic = Data(ind_select(j)).LFP.stability_halves.statistic; shuffles = Data(ind_select(j)).LFP.stability_halves.shuffles; pvalue = Data(ind_select(j)).LFP.stability_halves.pvalue;
                elseif i==3, statistic = Data(ind_select(j)).LFP.replayTrajCorr.statistic; shuffles = Data(ind_select(j)).LFP.replayTrajCorr.shuffles; pvalue = Data(ind_select(j)).LFP.replayTrajCorr.pvalue;
                elseif i==4, statistic = Data(ind_select(j)).LFP.ratTrajCorr.statistic; shuffles = Data(ind_select(j)).LFP.ratTrajCorr.shuffles; pvalue = Data(ind_select(j)).LFP.ratTrajCorr.pvalue;
                elseif i==5, statistic = Data(ind_select(j)).LFP.ratTrajCorr_immobility.statistic; shuffles = Data(ind_select(j)).LFP.ratTrajCorr_immobility.shuffles; pvalue = Data(ind_select(j)).LFP.ratTrajCorr_immobility.pvalue;
                elseif i==6, statistic = Data(ind_select(j)).LFP.rippleFieldPasses_mvl; shuffles = Data(ind_select(j)).LFP.rippleFieldPasses_mvl_shuffles; pvalue = Data(ind_select(j)).LFP.rippleFieldPasses_pvalue; shuffles(shuffles==1) = [];
                end
    
                % histogram(shuffles,edges{i},'edgecolor','none','facecolor','k')
                h = histcounts(shuffles,edges{i},'normalization','probability');
                plot(edges{i}(1:end-1) + mean(diff(edges{i}))/2,h,'k','linewidth',2)
                if i==1 && j==length(ind_select), ylabel('frac. of sessions'), end
                if j==length(ind_select), xlabel(names{i}), end
                set(gca,'fontsize',12), set(gcf,'color','w')
                pbaspect([4 1 1]), 
                xlim([min(edges{i}) max(edges{i})])
                yl = ylim;
                vline(statistic(1),'k:')
                text(statistic(1),yl(2),load_pstar(pvalue(1)),'fontsize',14,'HorizontalAlignment','left','VerticalAlignment','top')
            end
        end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot local vs global ripple power and PFs

    r = []; 
    for ii = 1:length(Data)  
        cd(strcat(Data(ii).ratName,'/processed/',Data(ii).file))
        load Experiment_Information
        load Analysis_Information
        cd ../../..
        sessionNum = Data(ii).sessionNum; sessionNum_decoder = sessionNum;

        LFP_mat = Data_LFP(ii).replays_amp_mat_tets;
        spikeDensity_mat = Data_spikeDensity(ii).replays_amp_mat_tets;

        ind_goodTets = find(Data(ii).tets_ripplePower>0.5 & Data(ii).tets_numCells>5);
        % ind_goodTets = find(Data(ii).tets_numCells>5);

        r_sub = nan(length(ind_goodTets),2);
        for i = 1:length(ind_goodTets)
            r_sub(i,1) = compute_corrCoef([Data_LFP(ii).replays_amp_mat(:) LFP_mat(ind_goodTets(i),:)']);
            r_sub(i,2) = compute_corrCoef([Data_spikeDensity(ii).replays_amp_mat(:) spikeDensity_mat(ind_goodTets(i),:)']);
        end
        r = [r; r_sub];

        % for j = 1:2
        %     figure(j)
        %     for i = 1:length(ind_goodTets)
        %         subplot(ceil(length(ind_goodTets)/3),3,i)
        %         rippleField_tet = reshape(LFP_mat(ind_goodTets(i),:),numSpatialBins);
        %         spikeDensityField_tet = reshape(spikeDensity_mat(ind_goodTets(i),:),numSpatialBins);
        % 
        %         imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        %         hold on
        %         if j==1, imagesc(compute_zscore_squareMatrix(rippleField_tet),'alphadata',~isnan(rippleField_tet)), caxis([0 2.5])
        %         else, imagesc(compute_zscore_squareMatrix(spikeDensityField_tet),'alphadata',~isnan(spikeDensityField_tet)), caxis([0 2.5])
        %         end
        %         hold off
        %         colorStyle = 'k'; plot_mazeProperties
        %         title(ind_goodTets(i),'fontsize',10)
        %     end
        % end
        % keyboard
        
    end
    
    histogram(r(:,1),linspace(-1,1,40),'facecolor','r','edgecolor','none'), set(gcf,'color','w')
    hold on, histogram(r(:,2),linspace(-1,1,40),'facecolor','k','edgecolor','none'), hold off
    axis square, xlabel('correlation'), ylabel('num. of tetrodes'), set(gca,'fontsize',14), legend('ripple','spike density','location','northwest')
    xlim([-0.5 1])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ripple fields vs place field sums (stable and unstable)
    colorStyle = 'k';
    for ii = 1:length(Data)  
        cd(strcat(Data(ii).ratName,'/processed/',Data(ii).file))
        load Experiment_Information
        load Analysis_Information
        load behavior
        cd ../../..
        sessionNum = Data(ii).sessionNum; sessionNum_decoder = sessionNum;

            replays = Data(ii).replays;
            replays_spikeDensity = Data(ii).spikeDensity.replays_amp;
            replays_LFP = Data(ii).LFP.replays_amp;
            
            replays_locs = compute_binsToLocs(replays(:,2:3),x_edges,y_edges);
            replayDensity_mat = hist3([replays_locs(:,2) replays_locs(:,1)],'Edges',{y_edges x_edges}); replayDensity_mat = replayDensity_mat(1:end-1,1:end-1);
                
            numSpatialBins_smoothing = 8;
            gaussFilt = setUp_gaussFilt(numSpatialBins,numSpatialBins_smoothing/binSize);

            [replays_mat_smoothed_LFP,replays_mat_smoothed_NaN_LFP] = compute_ampField(replays(:,2:3),replays_LFP(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
            [replays_mat_smoothed_spikeDensity,replays_mat_smoothed_NaN_spikeDensity] = compute_ampField(replays(:,2:3),replays_spikeDensity(:,end),replayDensity_mat,numSpatialBins,gaussFilt);

            Data_LFP(ii).replays_amp_mat = replays_mat_smoothed_NaN_LFP;
            Data_spikeDensity(ii).replays_amp_mat = replays_mat_smoothed_NaN_spikeDensity;

            if sum(~isnan(Data(ii).barrierDiff(:)))==0, continue, end

        subplot(1,5,1), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc((Data_LFP(ii).replays_amp_mat),'alphadata',~isnan(Data_LFP(ii).replays_amp_mat)), hold off
            plot_mazeProperties, title('ripple field'), caxis([0 1.5])
        
        subplot(1,5,2), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc((Data_spikeDensity(ii).replays_amp_mat),'alphadata',~isnan(Data_spikeDensity(ii).replays_amp_mat)), hold off
            plot_mazeProperties, title('spike density field'), caxis([0 2])
        
        subplot(1,5,3), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc(compute_zscore_squareMatrix(Data(ii).rateMapSum),'alphadata',~isnan(Data(ii).rateMapSum)), hold off
            plot_mazeProperties, title('place field sum'), caxis([-1 2])
        
        subplot(1,5,4), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc(compute_zscore_squareMatrix(Data(ii).rateMapSum_stable),'alphadata',~isnan(Data(ii).rateMapSum_stable)), hold off
            plot_mazeProperties, title('place field sum - stable'), caxis([-1 2])
        
        subplot(1,5,5), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc(compute_zscore_squareMatrix(Data(ii).rateMapSum_unstable),'alphadata',~isnan(Data(ii).rateMapSum_unstable)), hold off
            plot_mazeProperties, title('place field sum - unstable'), caxis([-1 2])
 
        set(findobj(gcf,'type','axes'),'FontSize',10), set(gcf,'color','w')
        keyboard
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot ripple events on top of ripple fields

    ind_select = [5,9,21,33];
    for ii = 1:length(Data)
        % if ~ismember(ii,ind_select), continue, end
        
        cd(strcat(Data(ii).ratName,'/processed/',Data(ii).file))
        load Experiment_Information
        load Analysis_Information
        sessionNum = Data(ii).sessionNum; sessionNum_decoder = sessionNum;
        cd ../../..
               
        replays = Data(ii).replays;
        replays_LFP = Data(ii).LFP.replays_amp;
        rippleField = Data(ii).LFP.replays_amp_mat;

        %ripple events
        eventZscoreThr = 2;
        eventDurationThr = 0.015;
        eventMergeThr = 0;
        rippleEvents = compute_events_times([replays_LFP(:,1),replays_LFP(:,end)],eventZscoreThr,eventMergeThr,eventDurationThr);
        rippleTimes = rippleEvents(:,3);

        imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
        hold on, imagesc(rippleField,'AlphaData',~isnan(rippleField)), hold off
        replays_rippleTime = compute_dataInterpolation(replays,rippleTimes);
        hold on, plot(replays_rippleTime(:,2),replays_rippleTime(:,3),'r*'), hold off
        plot_mazeProperties
        caxis([0 2])
        keyboard

    end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ind = find([Data_LFP_spatialInfo.pvalue]<0.05);
    data_replayDensity_stability_halves = [Data.replayDensity_stability_halves]';
    data_LFP_stability_halves = [Data_LFP_stability_halves.statistic]';
    data_spikeDensity_stability_halves = [Data_spikeDensity_stability_halves.statistic]';

    numBins = 16;
    edges = linspace(0,1,numBins); centers = edges(1:end-1) + mean(diff(edges));
    h1 = histcounts(data_replayDensity_stability_halves(ind),edges,'normalization','probability');
    h2 = histcounts(data_LFP_stability_halves(ind),edges,'normalization','probability');
    h3 = histcounts(data_spikeDensity_stability_halves(ind),edges,'normalization','probability');
    plot(centers,h1,'k','linewidth',3), hold on, plot(centers,h2,'r','linewidth',3), plot(centers,h3,'b','linewidth',3)
    legend('replay occ.','ripple field','spike density field')
    xlabel('spatial correlation'), ylabel('frac. of sessions')
    axis square, set(gca,'fontsize',16), set(gcf,'color','w')

    [~,pvalue] = ttest(data_replayDensity_stability_halves(ind),data_LFP_stability_halves(ind));
    [~,pvalue] = ttest(data_replayDensity_stability_halves(ind),data_spikeDensity_stability_halves(ind));




bar([nanmean(data_replayDensity_stability_halves(ind)) nanmean(data_LFP_stability_halves(ind)) nanmean(data_spikeDensity_stability_halves(ind))])



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Non-replay ripples
% 
% 
%     load replayEvents_SWRfield_3
% 
%     numShuffles = 100; 
%     for ii = 1:length(Data) 
%         ii
% 
%         cd(strcat(Data(ii).ratName,'/processed/',Data(ii).file))
%         load Experiment_Information
%         load Analysis_Information
%         load behavior
%         load clusters
%         load replayEvents
%         sessionNum = Data(ii).sessionNum; sessionNum_decoder = sessionNum;
%         load(strcat('LFP_SWRAmp_mean_',num2str(sessionNum),'.mat'))
%         load mazeTimes
%         lickingTimes = load_fieldVec([session(sessionNum).trials],'lickingTimes',2);
%         load binDecoding_08
%         load spikeDensity
%         cd ../../..
% 
%             numSpatialBins_smoothing = 8;
%             gaussFilt = setUp_gaussFilt(numSpatialBins,numSpatialBins_smoothing/binSize);
% 
%             %load replays
%                 replay_dispersionThr = 12; %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%                 load_ind_replayEvents
%                 [replays,replays_NaN,replays_NaNremoved,replays_interp,~,~,~,~,replays_smoothed] = compute_replaysConcatenation(decoder_replay(sessionNum),ind_replay);
%                 replays = replays_NaNremoved;
%                 replays_NaNremoved = replays; replays_NaNremoved(isnan(replays_NaNremoved(:,2)),:) = [];
%                     ind_NaN = find(isnan(replays(:,2)));
% 
%                 %replay position density
%                 replays_locs = compute_binsToLocs(replays(:,2:3),x_edges,y_edges);
%                 replayDensity_mat = hist3([replays_locs(:,2) replays_locs(:,1)],'Edges',{y_edges x_edges}); replayDensity_mat = replayDensity_mat(1:end-1,1:end-1);
%                 [replayDensity_mat_smoothed,replayDensity_mat_smoothed_NaN] = compute_smoothedPositionHist(replayDensity_mat,gaussFilt,replayDensity_mat);
% 
%             %SWRs
%                 ripplePower = LFP_SWRAmp_mean;
%                 ripplePower_session = compute_dataTemporalConcatenation(ripplePower,load_timeBounds(Run_Times{sessionNum}));
%                 lickingOnly_LFP = compute_dataTemporalConcatenation(ripplePower_session,lickingTimes);
%                     mean_LFP = nanmean(lickingOnly_LFP(:,2));
%                     std_LFP = nanstd(lickingOnly_LFP(:,2));
%                 ripplePower_session(:,3) = (ripplePower_session(:,2)-mean_LFP)/std_LFP;
%                 replays_LFP = compute_dataInterpolation(ripplePower_session,replays(:,1));
% 
%                 %compute field
%                 [replays_mat_smoothed_LFP,replays_mat_smoothed_NaN_LFP] = compute_ampField(replays(:,2:3),replays_LFP(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
% 
%                 %ripple fields from non-replay ripples
%                     eventZscoreThr = 3;
%                     eventDurationThr = 0.01;
%                     eventMergeThr = 0.05;
%                     rippleEvents = compute_events_times([ripplePower_session(:,1) ripplePower_session(:,3)],eventZscoreThr,eventMergeThr,eventDurationThr);
%                     ripplePeakTimes = rippleEvents(:,3); 
%                     replayTimes = timePoints(ind_replay,:);
% 
%                     %non-replay ripple times and ripple power
%                     [~,x] = compute_dataTemporalConcatenation([ripplePeakTimes ones(size(ripplePeakTimes))],replayTimes);
%                     ind_ripples_nonReplay = find(isnan(x(:,2)));
%                     rippleTimes_nonReplay = rippleEvents(ind_ripples_nonReplay,1:2);
%                     ripplePower_rippleTimes_nonReplay = compute_dataTemporalConcatenation(ripplePower_session,rippleTimes_nonReplay);
% 
%                     %decoded content during non-replay ripples
%                     [x_immobility_bins,~,~,posteriorSpread,~,~,~,posteriorPeak,~,timeBins_NaNremoved_merge] = compute_filtering_binDecoding(decoder_binDecoding(sessionNum),sequence_posteriorSpreadThr,[0 speedThr],sequence_spikeDensityThr,positions,spikeDensity,spikeDensityStepSize,sequence_jumpThr,load_timeBounds(Run_Times{sessionNum}),binSize);                    
%                     x_immobility_bins_rippleTimes_nonReplay = compute_dataInterpolation(x_immobility_bins,ripplePower_rippleTimes_nonReplay(:,1));
% 
%                     %non-replay ripple decoded position density
%                     x_immobility_locs_rippleTimes_nonReplay = compute_binsToLocs(x_immobility_bins_rippleTimes_nonReplay(:,2:3),x_edges,y_edges);
%                     ripples_nonReplays_Density_mat = hist3([x_immobility_locs_rippleTimes_nonReplay(:,2) x_immobility_locs_rippleTimes_nonReplay(:,1)],'Edges',{y_edges x_edges}); ripples_nonReplays_Density_mat = ripples_nonReplays_Density_mat(1:end-1,1:end-1);
% 
%                     %compute non-replay ripple field
%                     [~,ripples_nonReplays_mat_smoothed_NaN_LFP] = compute_ampField(x_immobility_bins_rippleTimes_nonReplay(:,2:3),ripplePower_rippleTimes_nonReplay(:,end),ripples_nonReplays_Density_mat,numSpatialBins,gaussFilt);
%                     [~,ripples_nonReplays_mat_smoothed_NaN_LFP_nonZscored] = compute_ampField(x_immobility_bins_rippleTimes_nonReplay(:,2:3),ripplePower_rippleTimes_nonReplay(:,end-1),ripples_nonReplays_Density_mat,numSpatialBins,gaussFilt);
% 
%                     %ripple field correlation
%                     corr_LFP = compute_corrCoef([ripples_nonReplays_mat_smoothed_NaN_LFP(:) replays_mat_smoothed_NaN_LFP(:)]);
% 
%                     %compute spatial information
%                     spatialInfo_LFP = compute_spatialInfo(ripples_nonReplays_mat_smoothed_NaN_LFP_nonZscored,ripples_nonReplays_Density_mat);
% 
%                     %compute stability
%                     LFP_mat_vec = nan(prod(numSpatialBins),2);
%                     for t = 1:2
%                         if t==1 %halves
%                             edges_sub = rippleTimes_nonReplay((1:round(size(rippleTimes_nonReplay,1)/2)),:);
%                         elseif t==2
%                             edges_sub = rippleTimes_nonReplay((round(size(rippleTimes_nonReplay,1)/2)+1:end),:);
%                         end
% 
%                         %non-replay ripple position density
%                             x_immobility_bins_rippleTimes_nonReplay_sub = compute_dataTemporalConcatenation(x_immobility_bins_rippleTimes_nonReplay,edges_sub);
%                             x_immobility_locs_rippleTimes_nonReplay_sub = compute_binsToLocs(x_immobility_bins_rippleTimes_nonReplay_sub(:,2:3),x_edges,y_edges);
%                             ripples_nonReplays_Density_mat_sub = hist3([x_immobility_locs_rippleTimes_nonReplay_sub(:,2) x_immobility_locs_rippleTimes_nonReplay_sub(:,1)],'Edges',{y_edges x_edges}); ripples_nonReplays_Density_mat_sub = ripples_nonReplays_Density_mat_sub(1:end-1,1:end-1);
% 
%                         %non-replay ripple power
%                             ripplePower_rippleTimes_nonReplay_sub = compute_dataTemporalConcatenation(ripplePower_rippleTimes_nonReplay,edges_sub);
%                                 L = min(size(ripplePower_rippleTimes_nonReplay_sub,1),size(x_immobility_bins_rippleTimes_nonReplay_sub,1));
%                                 ripplePower_rippleTimes_nonReplay_sub = ripplePower_rippleTimes_nonReplay_sub(1:L,:);
%                                 x_immobility_bins_rippleTimes_nonReplay_sub = x_immobility_bins_rippleTimes_nonReplay_sub(1:L,:);
%                             [ripples_nonReplays_mat_smoothed_LFP_sub,ripples_nonReplays_mat_smoothed_NaN_LFP_sub] = compute_ampField(x_immobility_bins_rippleTimes_nonReplay_sub(:,2:3),ripplePower_rippleTimes_nonReplay_sub(:,end),ripples_nonReplays_Density_mat_sub,numSpatialBins,gaussFilt);
%                             LFP_mat_vec(:,t) = ripples_nonReplays_mat_smoothed_LFP_sub(:);
% 
%                         %load
%                             Data(ii).LFP.nonReplayRipples.stability_halves.window(t).replays_amp_mat = ripples_nonReplays_mat_smoothed_NaN_LFP_sub;
%                     end
%                     stability_halves_LFP = compute_corrCoef(LFP_mat_vec);
% 
%                     % subplot(121), imagesc(reshape(LFP_mat_vec(:,1),numSpatialBins))
%                     % subplot(122), imagesc(reshape(LFP_mat_vec(:,2),numSpatialBins))
% 
% 
%             %shuffles
%                 corr_LFP_shuffle = nan(numShuffles,1);
%                 spatialInfo_LFP_shuffle = nan(numShuffles,1);
%                 stability_halves_LFP_shuffle = nan(numShuffles,1);
%                 for i = 1:numShuffles
% 
%                     randShift = round(rand*size(replays_LFP,1));
%                     replays_shuffle_LFP = circshift(replays_LFP,randShift);
%                     replays_shuffle = replays; replays_shuffle(:,2:end) = circshift(replays_shuffle(:,2:end),randShift);
% 
%                     %compute ripple field
%                         [replays_mat_smoothed_shuffle_LFP,replays_mat_smoothed_NaN_shuffle_LFP] = compute_ampField(replays(:,2:3),replays_shuffle_LFP(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
% 
%                     %ripple field correlation
%                         corr_LFP_shuffle(i) = compute_corrCoef([ripples_nonReplays_mat_smoothed_NaN_LFP(:) replays_mat_smoothed_NaN_shuffle_LFP(:)]);
% 
%                     %shuffles for non-replay associated ripples
%                         randShift = round(rand*size(ripplePower_rippleTimes_nonReplay,1));
%                         ripplePower_rippleTimes_nonReplay_shuffle = ripplePower_rippleTimes_nonReplay;
%                         ripplePower_rippleTimes_nonReplay_shuffle(:,2:3) = circshift(ripplePower_rippleTimes_nonReplay_shuffle(:,2:3),randShift);
% 
%                         %compute non-replay ripple field
%                         [~,ripples_nonReplays_mat_smoothed_NaN_LFP_shuffle] = compute_ampField(x_immobility_bins_rippleTimes_nonReplay(:,2:3),ripplePower_rippleTimes_nonReplay_shuffle(:,end),ripples_nonReplays_Density_mat,numSpatialBins,gaussFilt);
% 
%                         %compute spatial information
%                         spatialInfo_LFP_shuffle(i) = compute_spatialInfo(ripples_nonReplays_mat_smoothed_NaN_LFP_shuffle,ripples_nonReplays_Density_mat);
% 
%                         %compute stability of non-replay ripple fields
%                             LFP_mat_vec_ = nan(prod(numSpatialBins),2);
%                             for t = 1:2
%                                 if t==1 %halves
%                                     edges_sub = rippleTimes_nonReplay((1:round(size(rippleTimes_nonReplay,1)/2)),:);
%                                 elseif t==2
%                                     edges_sub = rippleTimes_nonReplay((round(size(rippleTimes_nonReplay,1)/2)+1:end),:);
%                                 end
% 
%                                 %non-replay ripple position density
%                                     x_immobility_bins_rippleTimes_nonReplay_sub = compute_dataTemporalConcatenation(x_immobility_bins_rippleTimes_nonReplay,edges_sub);
%                                     x_immobility_locs_rippleTimes_nonReplay_sub = compute_binsToLocs(x_immobility_bins_rippleTimes_nonReplay_sub(:,2:3),x_edges,y_edges);
%                                     ripples_nonReplays_Density_mat_sub = hist3([x_immobility_locs_rippleTimes_nonReplay_sub(:,2) x_immobility_locs_rippleTimes_nonReplay_sub(:,1)],'Edges',{y_edges x_edges}); ripples_nonReplays_Density_mat_sub = ripples_nonReplays_Density_mat_sub(1:end-1,1:end-1);
% 
%                                 %non-replay ripple power
%                                     ripplePower_rippleTimes_nonReplay_shuffle_sub = compute_dataTemporalConcatenation(ripplePower_rippleTimes_nonReplay_shuffle,edges_sub);
%                                         L = min(size(ripplePower_rippleTimes_nonReplay_shuffle_sub,1),size(x_immobility_bins_rippleTimes_nonReplay_sub,1));
%                                         ripplePower_rippleTimes_nonReplay_shuffle_sub = ripplePower_rippleTimes_nonReplay_shuffle_sub(1:L,:);
%                                         x_immobility_bins_rippleTimes_nonReplay_sub = x_immobility_bins_rippleTimes_nonReplay_sub(1:L,:);
% 
%                                     [ripples_nonReplays_mat_smoothed_LFP_shuffle_sub,ripples_nonReplays_mat_smoothed_NaN_LFP_shuffle_sub] = compute_ampField(x_immobility_bins_rippleTimes_nonReplay_sub(:,2:3),ripplePower_rippleTimes_nonReplay_shuffle_sub(:,end),ripples_nonReplays_Density_mat_sub,numSpatialBins,gaussFilt);
%                                     LFP_mat_vec(:,t) = ripples_nonReplays_mat_smoothed_LFP_shuffle_sub(:);
% 
%                             end
%                             stability_halves_LFP_shuffle(i) = compute_corrCoef(LFP_mat_vec);
%                             % subplot(121), imagesc(reshape(LFP_mat_vec(:,1),numSpatialBins))
%                             % subplot(122), imagesc(reshape(LFP_mat_vec(:,2),numSpatialBins))
%                             % drawnow
%                 end
% 
%             %compute pvalues
%                 corr_LFP_pvalue = (1+nansum(corr_LFP_shuffle>corr_LFP))/(1+numShuffles);
%                 spatialInfo_LFP_pvalue = (1+nansum(spatialInfo_LFP_shuffle>spatialInfo_LFP))/(1+numShuffles);
%                 stability_halves_LFP_pvalue = (1+nansum(stability_halves_LFP_shuffle>stability_halves_LFP))/(1+numShuffles);
% 
%             %save data
% 
%                 Data(ii).LFP.nonReplayRipples.rippleFieldCorr.pvalue = corr_LFP_pvalue;
%                 Data(ii).LFP.nonReplayRipples.rippleFieldCorr.statistic = corr_LFP;
%                 Data(ii).LFP.nonReplayRipples.rippleFieldCorr.shuffles = corr_LFP_shuffle;
% 
%                 Data(ii).LFP.nonReplayRipples.spatialInfo.pvalue = spatialInfo_LFP_pvalue;
%                 Data(ii).LFP.nonReplayRipples.spatialInfo.statistic = spatialInfo_LFP;
%                 Data(ii).LFP.nonReplayRipples.spatialInfo.shuffles = spatialInfo_LFP_shuffle;
% 
%                 Data(ii).LFP.nonReplayRipples.stability_halves.pvalue = stability_halves_LFP_pvalue;
%                 Data(ii).LFP.nonReplayRipples.stability_halves.statistic = stability_halves_LFP;
%                 Data(ii).LFP.nonReplayRipples.stability_halves.shuffles = stability_halves_LFP_shuffle;
% 
%                 Data(ii).LFP.nonReplayRipples.replays_amp_mat = ripples_nonReplays_mat_smoothed_NaN_LFP;
% 
%     end
% 
%         save('replayEvents_SWRfield_3','Data','-append')
% 
        % load replayEvents_SWRfield_3
        % 
        % Data_LFP = [Data.LFP];
        % Data_LFP_nonReplayRipples = [Data_LFP.nonReplayRipples];
        % Data_LFP_nonReplayRipples_rippleFieldCorr = [Data_LFP_nonReplayRipples.rippleFieldCorr];
        % Data_LFP_nonReplayRipples_spatialInfo = [Data_LFP_nonReplayRipples.spatialInfo];
        % Data_LFP_nonReplayRipples_stability_halves = [Data_LFP_nonReplayRipples.stability_halves];
        % corr_LFP_pvalue = [[Data_LFP_nonReplayRipples_spatialInfo.pvalue]' [Data_LFP_nonReplayRipples_stability_halves.pvalue]' [Data_LFP_nonReplayRipples_rippleFieldCorr.pvalue]'];
        % corr_LFP_statistic = [[Data_LFP_nonReplayRipples_spatialInfo.statistic]' [Data_LFP_nonReplayRipples_stability_halves.statistic]' [Data_LFP_nonReplayRipples_rippleFieldCorr.statistic]'];
        % 
        % corr_LFP_shuffle_spatialInfo = [(Data_LFP_nonReplayRipples_spatialInfo.shuffle)]';
        % corr_LFP_shuffle_stability_halves = [(Data_LFP_nonReplayRipples_stability_halves.shuffle)]';
        % corr_LFP_shuffle_rippleFieldCorr = [(Data_LFP_nonReplayRipples_rippleFieldCorr.shuffle)]';
        % corr_LFP_shuffle = real([corr_LFP_shuffle_spatialInfo(:) corr_LFP_shuffle_stability_halves(:) corr_LFP_shuffle_rippleFieldCorr(:)]);
        % set(gcf,'color','w')
        % numBins = 20; 
        % edges = {linspace(0,0.3,numBins),linspace(-0.5,1,numBins),linspace(-0.5,1,numBins)};
        % 
        % names = {'spatial info.','stability','corr. with ripple field'};
        % for i = 1:3
        %     centers = edges{i}(1:end-1) + mean(diff(edges{i}));
        %     subplot(2,3,i), 
        %     histogram(corr_LFP_statistic(:,i),edges{i},'facecolor','k','edgecolor','none','normalization','probability'), 
        %     h_shuff = histcounts(corr_LFP_shuffle(:,i),edges{i},'normalization','probability');
        %     hold on, plot(centers,h_shuff,'k','linewidth',2), hold off
        %     axis square, xlabel(names{i}), ylabel('count'), set(gca,'fontsize',12), axis tight, vline(0.05,'k--'), box on, set(gca,'fontsize',14)
        % 
        %     xl = xlim; yl = ylim; 
        %     text(0.05+diff(xl)*0.2,yl(2)-diff(yl)*0.1,strcat(num2str(sum(corr_LFP_pvalue(:,i)<=0.05)),'/',num2str(sum(~isnan(corr_LFP_pvalue(:,i))))),'verticalAlignment','top','horizontalAlignment','left','fontsize',12,'color','k')
        % 
        %     subplot(2,3,3+i), 
        %     histogram(corr_LFP_pvalue(:,i),linspace(0,1,numBins),'facecolor','k','edgecolor','none','normalization','probability'), 
        %     axis square, xlabel('p value'), ylabel('count'), set(gca,'fontsize',12), axis tight, vline(0.05,'k--'), box on, set(gca,'fontsize',14)
        % end
        % 
        % 
        % ind = find(sum(corr_LFP_pvalue(:,1:2)<0.05,2)==2 & corr_LFP_pvalue(:,3)<0.1);
        % % ind = [5,6,21,33];
        % for ii = 1:2%37
        % 
        %     cd(strcat(Data(ind(ii)).ratName,'/processed/',Data(ind(ii)).file))
        %     load Experiment_Information
        %     load Analysis_Information
        %     sessionNum = Data(ind(ii)).sessionNum;
        %     cd ../../..
        % 
        %     subplot(2,2,2*ii-1)
        %     % subplot(121)
        %     imagesc(compute_zscore_squareMatrix(Data(ind(ii)).LFP.replays_amp_mat))
        %     plot_mazeProperties, caxis([-1 2])
        %     if ii==1, title('ripple field'), end
        % 
        %     subplot(2,2,2*ii)
        %     % subplot(122)
        %     imagesc(compute_zscore_squareMatrix(Data(ind(ii)).LFP.nonReplayRipples.replays_amp_mat))
        %     plot_mazeProperties, caxis([-1 2])
        %     if ii==1, title('non-replay ripple field'), end
        % 
        %     % corr_LFP_pvalue(ii,:)
        %     % keyboard
        % end





% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %plot ripple power vs num. of cells on tetrode
%     tets_ripplePower = [Data.tets_ripplePower]; tets_ripplePower = tets_ripplePower(:);
%     tets_numCells = [Data.tets_numCells]; tets_numCells = tets_numCells(:);
% 
%     [r,p] = compute_corrCoef([tets_numCells,tets_ripplePower]);
%     edges = linspace(0,15,10); centers = edges(1:end-1)+mean(diff(edges))/2;
%     data_mean = nan(length(edges)-1,1);
%     data_sem = nan(length(edges)-1,1);
%     for i = 1:length(edges)-1
%         ind = find(tets_numCells>edges(i) & tets_numCells<=edges(i+1));
%         data_mean(i) = nanmean(tets_ripplePower(ind));
%         data_sem(i) = nanstd(tets_ripplePower(ind));
%     end
%     plot(tets_numCells,tets_ripplePower,'r.')
%     hold on, shadedErrorBar(centers,data_mean,data_sem,'lineprops',{'k','markerfacecolor','k','linewidth',2})
%     axis square, xlabel('# clusters on tetrode'), ylabel('tetrode ripple power'), set(gca,'fontsize',14), set(gcf,'color','w')
%     xl = xlim; yl = ylim; text(xl(2),yl(2),strcat('r=',num2str(compute_round(r,100)),load_pstar(p)),'verticalAlignment','top','horizontalAlignment','right','fontsize',14)
% 
