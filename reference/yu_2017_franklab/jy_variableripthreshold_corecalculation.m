function [ripthresh_out,riphistpdf]=jy_variableripthreshold_corecalculation(input_powertrace,varargin)
% core calculation function for
%jy_variableripthreshold
% only requires the z scored powertrace
JY_animals=[];

animal_dir=[];
animal_name=[];

day=[];
target_epochs=[];


for option = 1:2:length(varargin)-1
    if isstr(varargin{option})
        switch(varargin{option})
            case 'JY_animals'
                JY_animals = varargin{option+1};
            case 'animal_dir'
                animal_dir = varargin{option+1};
            case 'animal_name'
                animal_name = varargin{option+1};
            case 'day'
                day = varargin{option+1};
            case 'target_epochs'
                target_epochs = varargin{option+1};
            otherwise
                error(['Option ',varargin{option},' unknown.']);
        end
    else
        error('Options must be strings, followed by the variable');
    end
end

%% first z score



%% get distribution of data

histbins=[-10:0.01:50];

% zscore
%input_powertrace=cellfun(@(x) zscore(x),input_powertrace,'uniformoutput',0);


rippletrace_hist_o=cellfun(@(x) histc(x,histbins),input_powertrace,'uniformoutput',0);

% smooth to find peak
rippletrace_hist=cellfun(@(x) smooth(x,11)',rippletrace_hist_o,'uniformoutput',0);

% find peak in distribution
[rippletrace_peak]=cellfun(@(x) find(x==max(x)),rippletrace_hist,'uniformoutput',0);
rippletrace_peak=cellfun(@(x) x(1),rippletrace_peak,'uniformoutput',0);
% the distribution to the left of the peak is used to estimate
% noise
% mirror around peak and
% generate empiracal pdf of gaussian noise
% use original to find bounds
ripthresh_emp=cellfun(@(x,z) [(abs(histbins(histbins<histbins(z)))+histbins(z))'+histbins(z) ...
    (x((histbins<histbins(z)))')],...
    rippletrace_hist_o,rippletrace_peak,'uniformoutput',0);

ripthresh_pdf_below=cellfun(@(x,z) [histbins(histbins<=histbins(z))' ...
    (x((histbins<=histbins(z)))')],...
    rippletrace_hist_o,rippletrace_peak,'uniformoutput',0);
ripthresh_emp_pdf=[ripthresh_pdf_below{1};ripthresh_emp{1}];
ripthresh_emp_pdf=sortrows(ripthresh_emp_pdf,[1]);
ripthresh_emp_pdf_histbins=ripthresh_emp_pdf(:,1);
ripthresh_emp_pdf=ripthresh_emp_pdf(:,2);

%% differen methods to get cut off

% ---genrate cdf and get value where 1-limitval of data points lie below the
% value
ripthresh_emp_cdf=cumsum(ripthresh_emp_pdf);
ripthresh_emp_cdf=ripthresh_emp_cdf./max(ripthresh_emp_cdf);
% get value >10 sd
limitval=10e-5;
% limitval=exp(-limitval.^2/2)/(sqrt(2*pi)*limitval);
%
ripthresh_emp1=ripthresh_emp_pdf_histbins(find(ripthresh_emp_cdf>=(1-limitval),1,'first')+1);

% ---find when mirrored peak is 0, that is the empiracally derived
% threshold
ripthresh_emp2=cellfun(@(x) x(find(x(:,2)==0,1,'last'),1),ripthresh_emp,'uniformoutput',1);

% ---fit gaussian and get sd
f=fit(ripthresh_emp_pdf_histbins,ripthresh_emp_pdf,'gauss1');
% figure
% plot(f,ripthresh_emp_pdf_histbins,ripthresh_emp_pdf)

% set cuttoff
sdcutoff=2.5758; % corresponds to p=0.01 2 tail test
%sdcutoff=1.96; % corresponds to p=0.05 2 tail test
ripthresh_emp3=sdcutoff*f.c1+f.b1;

% select threshold to use
% noise floor seems to be better
% gov noise not really gaussian
ripthresh_out=ripthresh_emp1;
riphistpdf = rippletrace_hist_o{1}/length(input_powertrace{1});

if 1
    figure; set(gcf,'units','inches');set(gcf,'position',[1 1 15 5]); set(gcf,'PaperPositionMode','auto');
    set(gcf, 'PaperSize', [15 5],'color','w'); hold on
    
    for iF=1:2
        axh=subplot(1,2,iF); hold on;
        
    lhndl1=plot(histbins,rippletrace_hist_o{1},'color','k','linewidth',2);
    lhndlg=plot(f,ripthresh_emp_pdf_histbins,ripthresh_emp_pdf);    
    lhndl2=line([ripthresh_emp1 ripthresh_emp1],[0.1 max(rippletrace_hist{1})*1.1],'color','g','linewidth',2);
    lhndl3=line([ripthresh_emp2 ripthresh_emp2],[0.1 max(rippletrace_hist{1})*1.1],'color','g','linewidth',2,'linestyle','--');
    lhndl4=line([ripthresh_emp3 ripthresh_emp3],[0.1 max(rippletrace_hist{1})*1.1],'color','g','linewidth',2,'linestyle',':');
    
    line(ones(1,2).*histbins(rippletrace_peak{1}),[0.1 max(rippletrace_hist{1})*1.1],'color',[0.75 0.75 0.75],'linewidth',2);
 
    if iF==1
    ylim([0 max(rippletrace_hist{1})*1.1])
    xlim([-2 4]);
       legend(axh,[lhndl1,lhndlg',lhndl2,lhndl3,lhndl4],...
        {'Data','Baseline est.','Gauss. fit',sprintf('Thresh <%.1dprctl',limitval),'Thresh. noise ceiling',...
        'Thresh. p=0.01'},'location','northeast')
     title(sprintf('%s day %i epochs %s\n immobility ripple consensus distribution: ripple threshold=%.3f sd',...
        animal_name,day,...
        strjoin(cellfun(@num2str,num2cell(target_epochs,1),'uniformoutput',0),' '),ripthresh_out));

  
    else
    ylim([1 max(rippletrace_hist{1})*1.1])
    xlim([-2 30]);
    set(gca,'yscale','log');
    end
%         
%     end
    xlabel('z');
    ylabel('Count');
      end
    
%     try
%     if ~isempty(JY_animals)
%         fig_name=sprintf('%s/Plot/%s_conszdistr_d%ie%s.pdf',...
%             animal_dir,animal_name(1:min(3,length(animal_name))),day,...
%             strjoin(cellfun(@num2str,num2cell(target_epochs,1),'uniformoutput',0),'_'));
%     else
%         fig_name=sprintf('/data14a/jai/Superlin/Plot/%s_conszdistr_d%ie%s.pdf',...
%             animal_name(1:min(3,length(animal_name))),day,...
%             strjoin(cellfun(@num2str,num2cell(target_epochs,1),'uniformoutput',0),'_'));
%     end
%     %export_fig(fig_name,'-pdf')
%     saveas(gcf,fig_name,'pdf')
%     close
%     catch
%         fprintf('Specify save location\m')
%     end
end
