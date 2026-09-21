function ag_extractconsensus2(animaldir, animalprefix, eventname, eventeegname, days, tetfilters, min_suprathresh_duration, nstd, varargin)
%function extractconsensus(directoryname, fileprefix, eventname, day, tetrode, min_suprathresh_duration, nstd, varargin)

% based on jai's version of kk_extractconsensus2
% improves upon older extractconsensus in these two ways:
%
%  % 1. Unlike before, normalise and smooths individual tetrodes eeg --before-- taking the mean across sites,
%           instead of smoothing after summing
%           Per jai, the median is used instead of the mean
%
%  % 2. instead of an arbitrary threshold, uses immobility times during all run sessions of that day to estimate gaussian noise of the consensus ripple band
%           and let that determine an appropriate threshold
%
%animaldir - example '/data99/user/animaldatafolder/', a folder
%                containing processed matlab data for the animal
%
%fileprefix	- folder name where the day's data is stored
%day		- the day(s) to process
%min_suprathresh_duration - the time (in sec) which the signal
%       must remain above threshold to be counted as as ripple; default 0.015 s.

%nstd		- the number of standard dev that ripple must be from mean to
%			be detextractconsensusected. Start with 2.
%           - if empty, nstd is calculated using noise estimation from data
%
%options	'stdev', stdev   sets the size of the standard deviation used to
%				allow direct comparison across epochs
%       	'baseline', b   sets the size of the baseline used to
%				allow direct comparison across epochs
%           'maxpeakval, m	- ripples with maximal peaks above this value
%				are exluded.  Use this avoid detecting noise
%				events. Default 1000
%           'calcbaselineestimate', calulcates and saves estimate of baseline (changes dramatically with velocity)
%           'immobripsonly', only output rip events that occur during immobility (same cutoff as for noise estimate)
%
% Outputs:
%ripples 	- structue with various fields, including the following which
%			describe each ripple.
%	starttime - time of beginning of ripple
%	endtime	  - time of end of ripple
%	maxthresh - the largest threshold in stdev units at which this ripple
%			would still be detected.

disp(['processing ' animalprefix])

stdev = 0;
baseline = 0;
savetrace = 0;
maxpeakval = 2000;
% define the standard deviation for the Gaussian smoother which we
% apply before thresholding (this reduces sensitivity to spurious
% flucutations in the ripple envelope)
smoothing_width = 0.004; % 4 ms
%smoothing_width = 0.008; % 8 ms
runimmobility_thresh = 1;
numTF = length(tetfilters);
calcbaselineestimate = 0;
immobripsonly = 0;
runepfilter = 'isequal($environment,''run'')';

% process varargin and overwrite default values 
if (~isempty(varargin))
    assign(varargin{:});
end

tetinfo = loaddatastruct(animaldir, animalprefix, 'tetinfo');
task = loaddatastruct(animaldir, animalprefix, 'task');

runeps_all = evaluatefilter(task,'isequal($environment,''goal'')');   


for d = days

    %% ---part 1: estimate baseline
    
    if runimmobility_thresh
        
        baseline_TF = nan(1,numTF);
        stdev_TF = nan(1,numTF);
        thresh_TF = nan(1,numTF);
        
        noise_baseline_TF = nan(1,numTF);
        noise_stdev_TF= nan(1,numTF);
        noise_thresh_TF = nan(1,numTF);
        
        runeps = unique(runeps_all(runeps_all(:,1) == d,2))';
        
        for TF = 1:length(tetfilters)
            
            tetfilter = tetfilters{TF};
            tetlist =  evaluatefilter(tetinfo{d},tetfilter);  
            tetlist = unique(tetlist(ismember(tetlist(:,1),runeps),2));
            if isempty(tetlist)
                disp(sprintf('no valid detecting tetrodes, d %d',d))
                continue
            end
            
            powervals = [];
            noisevals=[];
            noisevals_all=cell(max(runeps),1);
            
            for ep = runeps
                
                
                % load eeg data
                eventeeg = loadeegstruct(animaldir, animalprefix, eventeegname, d, ep, tetlist');
                
                % obtain times vector from one of the tetrodes as a reference
                eegtimesvec_ref = geteegtimes(eventeeg{d}{ep}{tetlist(1)});
                epoch_starttime = eegtimesvec_ref(1);
                epoch_endtime = eegtimesvec_ref(end);
                numsamples_refeeg = length(eventeeg{d}{ep}{tetlist(1)}.data(:,1));
                if numsamples_refeeg ~= length(eegtimesvec_ref)
                    keyboard
                end
                
                % sum the squared amplitudes from each tetrode
                amptet = nan(length(tetlist),length(eegtimesvec_ref));  % initialize
                figure;
                for tt = 1:length(tetlist)
                    tet = tetlist(tt);
                    if useenvelope==0
                        % take the amplitude
                        data = double(eventeeg{d}{ep}{tet}.data(:,1))';
                    else
                        % take the envelope
                        data = double(eventeeg{d}{ep}{tet}.data(:,3))';
                    end
                    
                    % if there are brief periods of wild noise (filtered amplitude exceeds maxval),
                    % set values to 0 as a patch fix and report
                    wildnoise = (abs(data) > maxpeakval);
                    if any(wildnoise)
                        ms_of_wildnoise = round(1000*sum(wildnoise)/1500);
                        if ms_of_wildnoise > 5000
                            % if more than 5 s of wild noise, don't report consensus events in this case -- abandon the epoch
                            disp(sprintf('anim %s day%dep%d tet %d: wild noise periods totalling %d millisec (if 1500 Hz) -- (!!) exceeds 5 s',...
                                animalprefix,d,ep,tet,ms_of_wildnoise))
                            disp('(!!!!!) excluding epoch because of this -- consider eliminating tet as detector!')
                            keyboard
                            continue
                        else
                            disp(sprintf('anim %s day%dep%d tet %d: wild noise periods totalling %d millisec (if 1500 Hz) excluded in threshold calc',...
                                animalprefix,d,ep,tet,ms_of_wildnoise))
                            % simply set violating values to zero
                            data(wildnoise) = 0;
                        end
                    end
                    
                    
                    if length(data) == numsamples_refeeg
                        curr_amptet= data;
                    else
                        % if mismatch in # of samples for this tetrode (different DSPs often start at slightly different times),
                        % 'interpolate' (really just quick way of appending NaNs on the deficient end)
                        eegtimesvec = geteegtimes(eventeeg{d}{ep}{tet});
                        interpdata = interp1(eegtimesvec,data,eegtimesvec_ref,'nearest')';
                        
                        
                        
                        curr_amptet = interpdata;
                    end
                    
                    
                    % smooth squared amplitude with a kernel, then take sqrt
                    Fs = eventeeg{d}{ep}{tet}.samprate;
                    kernel = gaussian(smoothing_width*Fs, ceil(8*smoothing_width*Fs));
                    curr_amptet = (smoothvect(curr_amptet, kernel));
                    
                    if normeachtet==1
                        curr_std=nanstd(curr_amptet);
                        curr_mean=nanmean(curr_amptet);
                        curr_amptet=(curr_amptet-curr_mean)./curr_std;
                    end
                    
                    amptet(tt,:)=curr_amptet;
                    
                    if 1
                        subplot(1,2,1); hold on;
                        plot(data(1:10000));
                        subplot(1,2,2); hold on;
                        plot(curr_amptet(1:10000))
                        
                    end
                    
                    
                end
                
                % smooth the summed squared amplitudes then take the square root
                mindur = round(min_suprathresh_duration * Fs);
                % take the sum
                %powertrace = sum(amptet,1);
                
                % take the median
                % median avoids one tetrode dominating power
                powertrace = nanmedian(amptet,1);
                powertrace_mean=mean(amptet,1);
                %                 powertrace_max=max(amptet,[],1);

                %% estimate continuous baseline level
                % remove large deviations
                powertrace_z=zscore(powertrace);
                powertrace_filt=powertrace_z<2;
                
                % original filter designed
                %              filter_05=designeegfilt(Fs,0,0.5);
                %              save('/home/jai/Src/Filter/Hz0_5.mat','filter_05')
                
                % load filter
                load('Hz0_5.mat');
                % filter with low pass filter to get baseline variation
                datafilt05 = filtfilt(filter_05, 1, ...
                    powertrace(powertrace_filt));

                % interploate back to all times
                % by filling the gaps where large deviations use to be
                powertrace_fill=interp1(eegtimesvec_ref(powertrace_filt),datafilt05,eegtimesvec_ref);
            

                if 0
                    figure; hold on;
                    plot(powertrace./max(powertrace),'r')
                    plot(powertrace_s./max(powertrace_s),'b')
                    legend({'mean','sum'})
                    
                    figure; hold on;
                    plot(zscore(powertrace(~isnan(powertrace))),'r')
                    plot(zscore(powertrace_s(~isnan(powertrace_s))),'b')
                    legend({'mean','sum'})
                    
                    figure; hold on;
                    plot(eegtimesvec_ref-min(eegtimesvec_ref),  meandiff,'k')
                    plot(eegtimesvec_ref-min(eegtimesvec_ref),  powertrace_max-powertrace,'k')
                    plot(eegtimesvec_ref-min(eegtimesvec_ref), powertrace,'r')
                    plot(eegtimesvec_ref-min(eegtimesvec_ref), powertrace_median,'k')
                end
                
                %% get immobility periods
                %kk_timefilterscript
                immobilefilter = {'kk_get2dstate', '($immobility == 1)','immobility_velocity',4,'immobility_buffer',0};
%                 if JY_animals==0
%                     % first identify immobility times in this epoch
%                     immobilefilter=immobile;
%                     
%                 else
%                     immobilefilter=immobile_JY;
%                     
%                 end
                immoperiods = kk_evaluatetimefilter(animaldir,animalprefix, ...
                    {immobilefilter} , [d ep]);
                immoperiods=immoperiods{d}{ep};
                
                immovec = logical(list2vec(immoperiods,eegtimesvec_ref));
                validinds = immovec & ~isnan(powertrace');
                % now filter for immobility times in this run epoch
                
                powervals = [ powervals   powertrace(validinds) ];
                noisevals=[noisevals powertrace_fill(validinds)];
                noisevals_all{ep}=powertrace_fill;
               
            end
            
            totalbaselinedur = round( length(powervals) / Fs);
            disp(sprintf('TF #%d: %d s of baseline data',TF,totalbaselinedur));

            
            % calculate mean and std of powertrace, to use in extractevents
            baseline_TF(TF) = mean(powervals);
            stdev_TF(TF) = std(powervals);
            
            
            
            % get threshold by estimating noise spread
            if isempty(nstd)
%                 thresh_TF(TF)=jy_variableripthreshold_corecalculation({powervals},'JY_animals',JY_animals,...
%                     'animal_dir',animaldir,...
%                     'animal_name',animalprefix,...
%                     'day',d,...
%                     'target_epochs',runeps);
                  thresh_TF(TF) = jy_variableripthreshold_corecalculation({powervals},'animal_dir',animaldir,'animal_name',animalprefix,'day',d,'target_epochs',runeps);
            else
                thresh_TF(TF) = baseline_TF(TF) + nstd * stdev_TF(TF);

            end

            
            
            
            % calculate mean and sd of baseline estimation
            % noise threshold is 2SD above mean
            noise_baseline_TF(TF) = mean(noisevals);
            noise_stdev_TF(TF) = std(noisevals);
            noise_thresh_TF(TF) = noise_baseline_TF(TF) + 2* noise_stdev_TF(TF);
            
            
            
            
            if any(~isreal([baseline stdev]))
                error('these should be real valued..')
            end
            
            % optional: plot power and threshold
            if 0
                figure
                %whitebg([0 0 0])
                plot(eegtimesvec_ref-eegtimesvec_ref(1),powertrace,'k')
                hold on
                plot([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),[baseline baseline],'b');
                plot([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),[thresh thresh],'r');
                
                figure
                %whitebg([0 0 0])
                plot(eegtimesvec_ref-eegtimesvec_ref(1),powertrace,'k')
                hold on
                plot([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),[baseline baseline],'b');
                plot([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),[thresh thresh],'r');
                
            end
            
            
        end
        
        
        
        
    end
    
    
    
    
    % Now, calculate the actual power trace + consensus events
    
    clear eventscons ev
    clear eventtrace tr
    
    if runimmobility_thresh
        eps_process= runeps;
    else
        eps_process=1:length(task{d});
    end
    
    for ep = eps_process
        
        if ep > length(tetinfo{d}) || ~isfield(task{d}{ep},'type') || strcmp(task{d}{ep}.type,'failed sleep')
            disp('task struct has one more epoch than in tetinfo (or aberrant .type) - disregarding this epoch')
            continue
        end
        
        
        for TF = 1:length(tetfilters)
            
            tetfilter = tetfilters{TF};
            
            % retrieve valid tetrodes to iterate through
            tetlist =  evaluatefilter(tetinfo{d}{ep},tetfilter)';
            
            if isempty(tetlist)
                disp(sprintf('no valid detecting tetrodes, d %d e %d',d,ep))
                ev.eventname = '';
                ev.nstd = [];
                ev.min_suprathresh_duration = [];
                ev.tetfilter = tetfilter;
                ev.tetlist = [];
                ev.starttime = [];
                ev.endtime = [];
                ev.maxthresh = [];
                eventscons{d}{ep}{TF} = ev;
                clear ev
                continue
            end
            if length(tetlist) < 3
                disp(sprintf('day %d ep %d: less than 3 tets!',d,ep))
            end
            
            % load eeg data
            eventeeg = loadeegstruct(animaldir, animalprefix, eventeegname, d, ep, tetlist);
            
            % obtain times vector from one of the tetrodes as a reference
            eegtimesvec_ref = geteegtimes(eventeeg{d}{ep}{tetlist(1)});
            epoch_starttime = eegtimesvec_ref(1);
            epoch_endtime = eegtimesvec_ref(end);
            numsamples_refeeg = length(eventeeg{d}{ep}{tetlist(1)}.data(:,1));
            if numsamples_refeeg ~= length(eegtimesvec_ref)
                keyboard
            end
            
            % sum the squared amplitudes from each tetrode
            amptet = nan(length(tetlist),length(eegtimesvec_ref));  % initialize
            for tt = 1:length(tetlist)
                tet = tetlist(tt);
                if useenvelope==0
                    % take the amplitude
                    data = double(eventeeg{d}{ep}{tet}.data(:,1))';
                else
                    % take the envelope
                    data = double(eventeeg{d}{ep}{tet}.data(:,3))';
                end
                % if there are brief periods of wild noise (filtered amplitude exceeds maxval),
                % set values to 0 as a patch fix and report
                wildnoise = (abs(data) > maxpeakval);
                if any(wildnoise)
                    ms_of_wildnoise = round(1000*sum(wildnoise)/1500);
                    if ms_of_wildnoise > 10000
                        % if more than 5 s of wild noise, don't report consensus events in this case -- abandon the epoch
                        % event
                        ev.eventname = '';
                        ev.nstd = [];
                        ev.min_suprathresh_duration = [];
                        ev.tetfilter = tetfilter;
                        ev.tetlist = [];
                        ev.starttime = [];
                        ev.endtime = [];
                        ev.maxthresh = [];
                        ev.mean_runimmo = [];
                        ev.stdev_runimmo = [];
                        % trace
                        tr.eventname = '';
                        tr.tetfilter = [];
                        tr.tetlist = [];
                        tr.eegtimesvec_ref = [];
                        tr.powertrace = [];
                        tr.timerange = [];
                        tr.samprate = [];
                        tr.mean_runimmo = [];
                        tr.stdev_runimmo = [];
                        disp(sprintf('anim %s day%dep%d tet %d: wild noise periods totalling %d millisec (if 1500 Hz) -- (!!) exceeds 10 s',...
                            animalprefix,d,ep,tet,ms_of_wildnoise))
                        disp('(!!!!!) excluding epoch because of this -- consider eliminating tet as detector!')
                        eventscons{d}{ep}{TF} = ev;
                        clear ev;
                        continue
                    else
                        disp(sprintf('anim %s day%dep%d tet %d: wild noise periods totalling %d millisec (if 1500 Hz) excluded in threshold calc',...
                            animalprefix,d,ep,tet,ms_of_wildnoise))
                        % simply set violating values to zero
                        data(wildnoise) = 0;
                    end
                end
                
                
                
                
                
                if length(data) == numsamples_refeeg
                    curr_amptet= data;
                else
                    % if mismatch in # of samples for this tetrode (different DSPs often start at slightly different times),
                    % 'interpolate' (really just quick way of appending NaNs on the deficient end)
                    eegtimesvec = geteegtimes(eventeeg{d}{ep}{tet});
                    interpdata = interp1(eegtimesvec,data,eegtimesvec_ref,'nearest')';
                    curr_amptet = interpdata;
                end
                
                
                % smooth squared amplitude with a kernel, then take sqrt
                Fs = eventeeg{d}{ep}{tet}.samprate;
                kernel = gaussian(smoothing_width*Fs, ceil(8*smoothing_width*Fs));
                curr_amptet = (smoothvect(curr_amptet, kernel));
                
                if normeachtet==1
                    curr_std=nanstd(curr_amptet);
                    curr_mean=nanmean(curr_amptet);
                    curr_amptet=(curr_amptet-curr_mean)./curr_std;
                end
                
                amptet(tt,:)=curr_amptet;
                
                if 0
                    figure; hold on;
                    plot(data,'g');
                    plot(curr_amptet,'k')
                    
                end
                
                
                
                
                
            end
            
            % smooth the summed squared amplitudes then take the square root
            mindur = round(min_suprathresh_duration * Fs);
            %powertrace = sum(amptet,1);
            powertrace = median(amptet,1);
            
          
            %powertrace_f=powertrace-powertrace_fill;
            if 0
                figure; hold on
                
                plot(eegtimesvec_ref-epoch_starttime,powertrace,'r')
                %plot(eegtimesvec_ref-epoch_starttime,powertrace_f,'k')
                plot(eegtimesvec_ref-epoch_starttime,powertrace_fill,'g')
                plot(eegtimesvec_ref-epoch_starttime,powertrace_mean,'c')
            end
            %powertrace=powertrace_f;
            
            if 0
                figure; hold on;
                plot(powertrace)
                plot(dataf,'r')
                plot(data_filt,'g')
            end
            
            
            %  Set the baseline and standard deviation
            if runimmobility_thresh
                % Run immobility  baseline
                baseline = baseline_TF(TF);
                stdev = stdev_TF(TF);
                thresh = thresh_TF(TF);
                
                noise_baseline = noise_baseline_TF(TF);
                noise_stdev = noise_stdev_TF(TF);
                noise_thresh = noise_thresh_TF(TF);
                
                powertrace_fill_z=(powertrace_fill-noise_baseline)./noise_stdev;
                
                
            else
                % All times within epoch basleine
                powertrace_values = powertrace(~isnan(powertrace));
                baseline = mean(powertrace_values);
                stdev = std(powertrace_values);
                thresh = baseline + nstd * stdev;
                if any(~isreal([baseline stdev]))
                    error('these should be real valued..')
                end
            end
            
            % optional: plot power and threshold
            if 0
                figure; hold on
                %whitebg([0 0 0])
                plot(eegtimesvec_ref-eegtimesvec_ref(1),powertrace,'k')
                hold on
                plot([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),[baseline baseline],'b');
                plot([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),[thresh thresh],'r');

            end
            
            % extract the events if this is a valid trace
            if (thresh >-5) && any(find(powertrace < baseline))
                disp(sprintf('%s day %d epoch %d ',animalprefix,d,ep))
                % some odd error within the extractevents .mex makes it lock up
                % sometimes if the powertrace is not doubled.. therefore down below I just
                % take the first half of the output
                % need to make sure all values are positive so add an
                % offset factor
                offsetfactor=10;
                tmpevent = extractevents([powertrace'; powertrace']+offsetfactor,...
                    thresh+offsetfactor, baseline+offsetfactor, 0, mindur, 0)';

                
                if mod(size(tmpevent,1),2)==0
                    lastind = size(tmpevent,1)/2;
                else
                    lastind = (size(tmpevent,1)-1)/2;
                end
                
                tmpevent=tmpevent(1:lastind,:);
                
                if adaptivefilt==1
                % remove all events with noise above 2SD
                tmpevent_noise=noisevals_all{ep}(tmpevent(:,1));
                tmpevent=tmpevent(tmpevent_noise<noise_thresh,:);
                end    
                
                startind = tmpevent(:,1);
                endind = tmpevent(:,2);
                
                % Event output
                ev.eventname = eventname;
                ev.nstd = nstd;
                ev.min_suprathresh_duration = min_suprathresh_duration;
                ev.tetfilter = tetfilter;
                ev.tetlist = tetlist;
                ev.starttime = epoch_starttime + startind / Fs;
                ev.endtime = epoch_starttime + endind / Fs;
                ev.threshold=thresh;
                
                if 1
                    figure; hold on
                    
                    otherrips=[ev.starttime ev.endtime]-epoch_starttime;
                    ttt=otherrips;
                    nspikes   = length(ttt); % number of elemebts / spikes
                    for nn = 1:nspikes % for every spike
                        jbfill([(ttt(nn,1)) (ttt(nn,2))],...
                            [10 10],[-2 -2],...
                            [250 220 150]./255,[],[],1);
                        %line(([tt(nn) tt(nn)+t(nn)]-mean(tstartdata))/10000,[0 length(spikes)+1],'Color','b','LineWidth',0.1);
                    end
                    
                    %whitebg([0 0 0])
                    plot(eegtimesvec_ref-eegtimesvec_ref(1),powertrace,'k')
                    hold on
                    plot(eegtimesvec_ref -eegtimesvec_ref(1),noisevals_all{ep},'r')
                     line([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),[noise_thresh noise_thresh],'color','g')
                     line([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),[thresh thresh],'color',[0.5 0.5 0.5])
                     line([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),...
                         [baseline baseline],'color','c')
                   %line([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),...
                        % [baseline baseline]-nstd*stdev,'color','c')
                     line([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),...
                         [baseline baseline]+2*stdev,'color',[0.5 0.5 0.5],'linestyle','--')
                        line([eegtimesvec_ref(1) eegtimesvec_ref(end)]-eegtimesvec_ref(1),...
                         [baseline baseline]+3*stdev,'color',[0.5 0.5 0.5],'linestyle',':')
                   figure
                      plot(eegtimesvec_ref(immovec)-eegtimesvec_ref(1),-1*ones(sum(immovec),1),'color',[1 0 0],'linestyle','-')
                end
                
                
                
                
                ev.maxthresh = (tmpevent(:,9) - baseline-offsetfactor) / stdev;
                if runimmobility_thresh
                    ev.mean_runimmo = baseline;
                    ev.stdev_runimmo = stdev;
                else
                    ev.mean_runimmo = [];
                    ev.stdev_runimmo = [];
                end
                ev.runimmobility_thresh = runimmobility_thresh;
                % Trace output
                tr.eventname = eventeegname;
                tr.tetfilter = tetfilter;
                tr.tetlist = tetlist;
                tr.eegtimesvec_ref = eegtimesvec_ref;
                tr.powertrace = powertrace;
                tr.timerange = [epoch_starttime epoch_endtime];
                tr.samprate = Fs;
                tr.threshold=[];
                if runimmobility_thresh
                    tr.mean_runimmo = baseline;
                    tr.stdev_runimmo = stdev;
                else
                    tr.mean_runimmo = [];
                    tr.stdev_runimmo = [];
                end
            else
                % Event output
                ev.eventname = '';
                ev.nstd = [];
                ev.min_suprathresh_duration = [];
                ev.tetfilter = '';
                ev.tetlist = [];
                ev.starttime = [];
                ev.endtime = [];
                ev.maxthresh = [];
                ev.mean_runimmo = [];
                ev.stdev_runimmo = [];
                ev.threshold=[];
                % Trace output
                tr.bandname = '';
                tr.tetfilter = [];
                tr.tetlist = [];
                tr.eegtimesvec_ref = [];
                tr.powertrace = [];
                tr.timerange = [];
                tr.samprate = [];
                tr.mean_runimmo = [];
                tr.stdev_runimmo = [];

                fprintf('%s day %d epoch %d no valid data for extraction\n',animalprefix,d,ep)
            end
            
            if any(find(powertrace<baseline))==0
                warning(['No below baseline values in data.  Fields left blank, day ' num2str(d) 'epoch ' num2str(ep)])
            end
            
            ev.timerange = [epoch_starttime epoch_endtime];
            ev.samprate = Fs;
            ev.baseline = baseline;
            ev.std = stdev;
            
            
            % for troubleshooting purposes can toggle this -- will make files
            % large though (and could make kk_getconstimes use a ton of memory)
            if 0
                ev.eegtimesvec_ref = eegtimesvec_ref;
                ev.powertrace = powertrace;
            end
            
            % Event
            clear eventeeg
            eventscons{d}{ep}{TF} = ev;
            clear ev;
            
            eventtrace{d}{ep}{TF} = tr;
            clear tr
            
        end
        
        % Trace output
        tracevarname = [eventeegname sprintf('trace%s%s%s',...
            normeachtettext{normeachtet+1},adaptivefilttext{adaptivefilt+1}),useenvelopetext{useenvelope+1}];
        if exist('eventtrace','var')
            eval([tracevarname ' = eventtrace']);
            save(sprintf('%s/CONS/%s%strace%s%s%s%02d-%d.mat', animaldir, animalprefix, eventeegname,...
                normeachtettext{normeachtet+1},adaptivefilttext{adaptivefilt+1},useenvelopetext{useenvelope+1},d,ep),tracevarname);
            clear eventtrace;
        else
            disp(sprintf('trace not saved for %s %s, day %d',animalprefix,bandname,d))
        end
        
    end
    
    
    
    consvariablename = [eventname sprintf('cons%s%s%s',...
        normeachtettext{normeachtet+1},adaptivefilttext{adaptivefilt+1}),useenvelopetext{useenvelope+1}];
    
    if exist('eventscons','var')
        eval([consvariablename ' = eventscons']);
        
        if isempty(masterfolder)
            savedir = animaldir;
        else
            savedir = masterfolder;
        end
        save(sprintf('%s/%s%scons%s%s%s%02d.mat', savedir, animalprefix, eventname,normeachtettext{normeachtet+1},...
            adaptivefilttext{adaptivefilt+1},useenvelopetext{useenvelope+1},d),consvariablename);
        cd(savedir)
        clear eventscons;
    else
        disp(sprintf('consensus not saved for %s %s, day %d',animalprefix,eventname,d))
    end
    
    
    
    
end

end
