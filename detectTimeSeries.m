clear
% load in some data
load d5331
dat=X;
X=dat(:,2:5); % 4 separate tetrode channels
%%  High Pass Filter the data
SamplingFreq=1e4;
cutoff=500;
[z,p,k]=butter(4,cutoff./SamplingFreq*2,'high');
[sos,g] = zp2sos(z,p,k);	     % Convert to SOS form
Hd = dfilt.df2tsos(sos,g);   % Create a dfilt object
X=filter(Hd,X);
% Normalize the data
X=bsxfun(@rdivide,bsxfun(@minus,X,mean(X)),std(X));
%% Whiten the filtered data:
ch=1; % which channel to use here with the tetrode.
if true % whether to whiten, changing to false will use original filtered data
    numLags=2; %Recommend a low number
    ac=xcov(X(:,ch),2*numLags,'unbiased');t=2*numLags+1;
    T=abs(bsxfun(@minus,-numLags:numLags,(-numLags:numLags)'));
    d=chol(ac(T+numLags*2+1))'\[zeros(numLags,1);1;zeros(numLags,1)];
    Xf=conv(X(:,ch),d,'same');
    Xf=bsxfun(@rdivide,bsxfun(@minus,Xf,mean(Xf)),std(Xf));
    % plot(xcov(Xf(:,ch),20,'unbiased'),'.');
else
    Xf=X(:,ch);
end
%% Threshold Detection
threshold=3;
windowlength=round(.001*SamplingFreq); %1ms window
[spikes,ts]=thresholdDetect(Xf,threshold,windowlength);
%% Run sorting algorithm:
K=5;
Sorter=FMM(spikes,K);
Sorter.align=false;
Sorter.FMMparam=1e-2; %% Changes how aggresively to cluster, range 0-1
Sorter.initialize;
Sorter.runVBfit;
%% Plots
fignumber1=1;
fignumber2=2;
Sorter.drawPCA3d(fignumber1,true)
Sorter.drawClusters(fignumber2)