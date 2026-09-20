clear all
cd analysis_RippleField/
% load replayEvents_rippleFieldStats
load replayEvents_rippleFieldStats_allCells_sigReplays
cd ..

numShuffles = 100;      

ind_comparison_1 = [1 2 3 4 5 6 11 12];
ind_comparison_2 = [12+17+3 12+17+4 12+10 12+11 12+13 12+14 12+16 12+17];
r_LFP = nan(length(ind_comparison_1),1);
r_spikeDensity = nan(length(ind_comparison_1),1);
r_LFP_shuffles = nan(length(ind_comparison_1),numShuffles);
r_spikeDensity_shuffles = nan(length(ind_comparison_1),numShuffles);
pvalues_LFP = nan(length(ind_comparison_1),1);
pvalues_spikeDensity = nan(length(ind_comparison_1),1);
for ii = 1:length(ind_comparison_1) 
    ii

    rippleField_1 = Data(ind_comparison_1(ii)).LFP.replays_amp_mat;
    rippleField_2 = Data(ind_comparison_2(ii)).LFP.replays_amp_mat;
    rippleField_1 = nanimresize(rippleField_1, size(rippleField_2));
    r_LFP(ii) = compute_corrCoef([rippleField_1(:) rippleField_2(:)]);

    spikeDensityField_1 = Data(ind_comparison_1(ii)).spikeDensity.replays_amp_mat;
    spikeDensityField_2 = Data(ind_comparison_2(ii)).spikeDensity.replays_amp_mat;
    spikeDensityField_1 = nanimresize(spikeDensityField_1, size(spikeDensityField_2));
    r_spikeDensity(ii) = compute_corrCoef([spikeDensityField_1(:) spikeDensityField_2(:)]);

    replays = Data(ind_comparison_2(ii)).replays;
    replays_LFP = Data(ind_comparison_2(ii)).LFP.replays_amp;
    replays_spikeDensity = Data(ind_comparison_2(ii)).spikeDensity.replays_amp;
    replayDensity_mat = Data(ind_comparison_2(ii)).replayDensity_mat;

    %histogram smoothing
        numSpatialBins = size(rippleField_2);
        numSpatialBins_smoothing = 8;
        binSize = 2;
        gaussFilt = setUp_gaussFilt(numSpatialBins,numSpatialBins_smoothing/binSize);

    %shuffles
        for i = 1:numShuffles
            randShift = round(rand*size(replays_LFP,1));
            replays_shuffle_LFP = [replays_LFP(:,1) circshift(replays_LFP(:,2:end),randShift)];
            replays_shuffle_spikeDensity = [replays_LFP(:,1) circshift(replays_spikeDensity(:,2:end),randShift)];
    
            %compute ripple and PBE fields 
                [~,rippleField_2_shuffle] = compute_ampField(replays(:,2:3),replays_shuffle_LFP(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
                [~,spikeDensityField_2_shuffle] = compute_ampField(replays(:,2:3),replays_shuffle_spikeDensity(:,end),replayDensity_mat,numSpatialBins,gaussFilt);
       
           r_LFP_shuffles(ii,i) = compute_corrCoef([rippleField_1(:) rippleField_2_shuffle(:)]);
           r_spikeDensity_shuffles(ii,i) = compute_corrCoef([spikeDensityField_1(:) spikeDensityField_2_shuffle(:)]);
        end
        pvalues_LFP(ii) = (1+nansum(r_LFP_shuffles(ii,:)>r_LFP(ii)))/(1+numShuffles);
        pvalues_spikeDensity(ii) = (1+nansum(r_spikeDensity_shuffles(ii,:)>r_spikeDensity(ii)))/(1+numShuffles);

end

cd analysis_RippleField/
save('replayEvents_rippleFieldStats_acrossRats.mat','r_LFP', 'r_spikeDensity', 'r_LFP_shuffles', 'r_spikeDensity_shuffles', 'pvalues_LFP', 'pvalues_spikeDensity')
cd ..

keyboard
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd analysis_RippleField/
load replayEvents_rippleFieldStats_acrossRats
cd ..

%stats
    edges = linspace(-0.8,1,12); centers = edges(1:end-1) + mean(diff(edges));
    
    set(gcf,'color','w')
    subplot(221)
    h = histcounts(r_LFP_shuffles(:),edges,'Normalization','probability');
    histogram(r_LFP,edges,'Normalization','probability','edgecolor','none','FaceColor','k')
    hold on, plot(centers,h,'k','linewidth',2), hold off
    xlabel('ripple field corr.'), ylabel('frac. of sessions')
    set(gca,'fontsize',14), axis square, xlim([min(edges) max(edges)])
    
    subplot(222)
    histogram(pvalues_LFP,linspace(0,1,20),'Normalization','probability','edgecolor','none','FaceColor','k')
    xlabel('pvalues'), ylabel('frac. of sessions')
    set(gca,'fontsize',14), axis square, 
    vline(0.05,'k--')
    xl = xlim; yl = ylim;
    [X2, p_combined] = fishers_combined_method(pvalues_LFP);
    text(0.05+diff(xl)*0.1,yl(2)-diff(yl)*0.1,strcat(num2str(nansum(pvalues_LFP<=0.05)),'/',num2str(sum(~isnan(pvalues_LFP))),load_pstar(p_combined)),'verticalAlignment','top','horizontalAlignment','left','fontsize',12)
    
    subplot(223)
    h = histcounts(r_spikeDensity_shuffles(:),edges,'Normalization','probability');
    histogram(r_spikeDensity,edges,'Normalization','probability','edgecolor','none','FaceColor','k')
    hold on, plot(centers,h,'k','linewidth',2), hold off
    xlabel('spike density field corr.'), ylabel('frac. of sessions')
    set(gca,'fontsize',14), axis square, xlim([min(edges) max(edges)])
    
    subplot(224)
    histogram(pvalues_spikeDensity,linspace(0,1,20),'Normalization','probability','edgecolor','none','FaceColor','k')
    xlabel('pvalues'), ylabel('frac. of sessions')
    set(gca,'fontsize',14), axis square, 
    vline(0.05,'k--')
    xl = xlim; yl = ylim;
    [X2, p_combined] = fishers_combined_method(pvalues_spikeDensity);
    text(0.05+diff(xl)*0.1,yl(2)-diff(yl)*0.1,strcat(num2str(nansum(pvalues_spikeDensity<=0.05)),'/',num2str(sum(~isnan(pvalues_spikeDensity))),load_pstar(p_combined)),'verticalAlignment','top','horizontalAlignment','left','fontsize',12)


%Plot examples
    cd analysis_RippleField/
    % load replayEvents_rippleFieldStats
    load replayEvents_rippleFieldStats_allCells_sigReplays
    cd ..

%%
    % ind_comparison_1 = [1 2 3 4 5 6 11 12];
    % ind_comparison_2 = [12+17+3 12+17+4 12+10 12+11 12+13 12+14 12+16 12+17];
    ind_comparison_1 = [1 2  5 6];
    ind_comparison_2 = [12+17+3 12+17+4 12+13 12+14];
    pvalues_LFP_sub = pvalues_LFP([1 2 5 6]);
    pvalues_spikeDensity_sub = pvalues_LFP([1 2 5 6]);
    for ii = 1:length(ind_comparison_1) 
        ii

        figure
        cd(strcat(Data(ind_comparison_1(ii)).ratName,'/processed/',Data(ind_comparison_1(ii)).file))
        load Experiment_Information
        load Analysis_Information
        load behavior
        cd ../../..
        sessionNum = Data(ind_comparison_1(ii)).sessionNum; sessionNum_decoder = sessionNum;

        subplot(2,4,1), 
            plot(Data(ind_comparison_1(ii)).replays(:,2),Data(ind_comparison_1(ii)).replays(:,3),'color',0.3*ones(1,3)), 
            hold on, plot(Data(ind_comparison_1(ii)).ratLocs(:,1),Data(ind_comparison_1(ii)).ratLocs(:,2),'r.','markersize',12); hold off
            axis square, colorStyle = 'k'; plot_mazeProperties

        subplot(2,4,2), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc((Data(ind_comparison_1(ii)).LFP.replays_amp_mat),'alphadata',~isnan(Data(ind_comparison_1(ii)).LFP.replays_amp_mat)), hold off
            % hold on, imagesc((Data(ind_comparison_1(ii)).spikeDensity.replays_amp_mat),'alphadata',~isnan(Data(ind_comparison_1(ii)).spikeDensity.replays_amp_mat)), hold off
            axis square, colorStyle = 'k'; plot_mazeProperties, caxis([0 1.5])

        subplot(2,4,6), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc((Data(ind_comparison_1(ii)).replayDensity_smoothed_mat),'alphadata',~isnan(Data(ind_comparison_1(ii)).replayDensity_smoothed_mat)), hold off
            axis square, colorStyle = 'k'; plot_mazeProperties, %caxis([0 2])

        cd(strcat(Data(ind_comparison_2(ii)).ratName,'/processed/',Data(ind_comparison_2(ii)).file))
        load Experiment_Information
        load Analysis_Information
        load behavior
        cd ../../..
        sessionNum = Data(ind_comparison_2(ii)).sessionNum; sessionNum_decoder = sessionNum;

        subplot(2,4,3), 
            plot(Data(ind_comparison_2(ii)).replays(:,2),Data(ind_comparison_2(ii)).replays(:,3),'color',0.3*ones(1,3)), 
            hold on, plot(Data(ind_comparison_2(ii)).ratLocs(:,1),Data(ind_comparison_2(ii)).ratLocs(:,2),'r.','markersize',12); hold off
            axis square, colorStyle = 'k'; plot_mazeProperties
        
        subplot(2,4,4), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc((Data(ind_comparison_2(ii)).LFP.replays_amp_mat),'alphadata',~isnan(Data(ind_comparison_2(ii)).LFP.replays_amp_mat)), hold off
            % hold on, imagesc((Data(ind_comparison_2(ii)).spikeDensity.replays_amp_mat),'alphadata',~isnan(Data(ind_comparison_2(ii)).spikeDensity.replays_amp_mat)), hold off
            axis square, colorStyle = 'k'; plot_mazeProperties, caxis([0 1.5])

        subplot(2,4,8), 
            imagesc(0.3*ones(numSpatialBins(1),numSpatialBins(2),3))
            hold on, imagesc((Data(ind_comparison_2(ii)).replayDensity_smoothed_mat),'alphadata',~isnan(Data(ind_comparison_2(ii)).replayDensity_smoothed_mat)), hold off
            axis square, colorStyle = 'k'; plot_mazeProperties, %caxis([0 1.5])


            rippleField_1 = imresize(Data(ind_comparison_1(ii)).spikeDensity.replays_amp_mat,[50 50]);
            rippleField_2 = imresize(Data(ind_comparison_2(ii)).spikeDensity.replays_amp_mat,[50 50]);
    
            subplot(2,4,4)
            r = compute_corrCoef([rippleField_1(:) rippleField_2(:)]);
            title(strcat('r=',num2str(compute_round(r,100)),load_pstar(pvalues_spikeDensity_sub(ii))))
            r

            posDensity_1 = imresize(Data(ind_comparison_1(ii)).replayDensity_smoothed_mat,[50 50]);
            posDensity_2 = imresize(Data(ind_comparison_2(ii)).replayDensity_smoothed_mat,[50 50]);
    
            subplot(2,4,8)
            r = compute_corrCoef([posDensity_1(:) posDensity_2(:)]);
            title(strcat('r=',num2str(compute_round(r,100))))
            r

    end
%%