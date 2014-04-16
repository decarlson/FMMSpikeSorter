function [spikes,timestamps]=thresholdDetect(xp,thres,spikelength)
if nargin<2
    thres=3*std(xp);
end
if nargin<3
    P=20;
else
    P=spikelength;
end
x=xp;
N=numel(xp);
maxpoint=round(.5*P);
timepoints=[];
for t=2*P+1:P:N-P
    wind=xp(t-P:t+P);
    [val,ndx]=max(wind);
    if val>thres
        if ndx<P*3/2;
            timepoints=[timepoints,t-P-1+ndx];
            xp(timepoints(end)+(-P:P))=0;
        end
    end
end
spikes=zeros(P,numel(timepoints));
for t=1:numel(timepoints)
    spikes(:,t)=x(timepoints(t)+(-maxpoint+1:P-maxpoint));
end
timestamps=timepoints;