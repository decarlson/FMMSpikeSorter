clear
example1=true;
example2=false;
K=3;
if example1
    figure(1);clf
    load clusters
    Sorter=FMM(Spike,K);
end
if example2
    figure(1);clf
    load HarrisData
    Sorter=FMM(squeeze(ec_spikes(1,:,:)));
end
%% Whether to align data:
Sorter.align=false;
%% initialize object
Sorter.initialize;
%% Try Sampler
Sorter.runMCMCsampler;
z=Sorter.getMAPassignment;
fignumber1=1;
fignumber2=2;
Sorter.drawPCA(fignumber1,true)
Sorter.drawClusters(fignumber2)
%% Try VB
Sorter.runVBfit;
fignumber1=3;
fignumber2=4;
Sorter.drawPCA3d(fignumber1,true)
Sorter.drawClusters(fignumber2)