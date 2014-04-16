classdef FMM<handle
    properties(GetAccess = 'public', SetAccess = 'public')
        % define the properties of the class here, (like fields of a struct)
        waveforms=[];
        K=[];
        x=[];
        maxClusters=20;
        collectionSamples=100;
        space=10;
        burnin=0;
        maxiter=[];
        llk=[];
        focused=true;
        maxllk=-inf;
        alpha=.1;
        drawing=true;
        align=false;
        FMMparam=1e-2;
    end
    properties(GetAccess = 'private', SetAccess ='public')
        iter=[];
        colors=[];
        muu0=[];
        lam0=[];
        v0=[];
        sig0=[];
        muu=[];
        prec={};
        sigma={};
        pii=[];
        lpii=[];
        z=[];
        zmap=[];
        ngam=[];
        qz=[];
        
        drawingFreq=30;
        alphaHigherLevel=[];
    end
    methods
        %% Method Functions
        % methods, including the constructor are defined in this block
        function obj = FMM(wf,K)
            % class constructor
            if(nargin > 0)
                obj.waveforms=wf;
            end
            if(nargin>1)
                obj.K=K;
            else
                obj.K=3;
            end
            obj.maxiter=obj.space*obj.collectionSamples+obj.burnin;
            obj.muu0=zeros(obj.K,1);
            obj.lam0=.1;
            obj.v0=obj.K+2;
            obj.sig0=eye(obj.K);
            obj.colors=jet(obj.maxClusters);
            if obj.focused
                obj.alphaHigherLevel=gamrnd(obj.alpha,1,obj.maxClusters,1);
            end
        end
        function initialize(obj)
            obj.muu=randn(obj.K,obj.maxClusters);
            obj.prec=cell(obj.maxClusters,1);
            for n=1:obj.maxClusters
                obj.prec{n}=eye(obj.K);
            end
            [obj.pii,obj.lpii]=drchrnd(obj.alpha,[],obj.maxClusters);
            if obj.align
                obj.wfalign()
            else
                [u,s,v]=svd(obj.waveforms,'econ');
                t=v(:,1:obj.K);
                t=bsxfun(@minus,t,mean(t));
                t=bsxfun(@rdivide,t,std(t));
                obj.x=t';
            end
            
        end
        function z=getMAPassignment(obj)
            z=obj.zmap;
            nz=zeros(obj.maxClusters,1);
            for a=1:numel(nz)
                nz(a)=sum(z==a);
            end
            [nz,rendx]=sort(nz,'descend');
            z2=zeros(size(z));
            for a=1:numel(nz);
                z2(z==rendx(a))=a;
            end
            obj.z=z2;
            z=z2;
  
            
        end
        
        function runVBfit(obj)
            % [obj.z,obj.qz,obj.mu,obj.sigma]=DPmergefit(obj.x',obj.maxClusters,obj.alph,k0,v0,muu0,phi0,FMM);
            [obj.z,obj.qz,obj.muu,obj.sigma]=DPmergefit(obj.x,obj.maxClusters,obj.alpha,obj.lam0,obj.v0,obj.muu0,obj.sig0,obj.FMMparam*size(obj.x,2));
            obj.zmap=obj.z;
        end
        
        function runMCMCsampler(obj)
            for iter=1:obj.maxiter
                obj.iter=iter;
                MCMCsample(obj)
                if obj.drawing
                    if mod(iter,obj.drawingFreq)==0
                        obj.drawPCA;
                    end
                end
                if ((obj.iter>obj.burnin) & mod(obj.iter-obj.burnin,obj.space))==0
                    obj.llk=GMMloglikelihood(obj.x',obj.z,obj.muu',obj.prec,obj.pii);
                    if obj.llk>obj.maxllk
                        obj.maxllk=obj.llk;
                        obj.zmap=obj.z;
                    end
                end
            end
        end
        function MCMCsample(obj)
            obj.sampleAssignments;
            obj.sampleAtoms;
            obj.sampleWeights;
        end
        function sampleAssignments(obj)
            logGaussianLikelihood=GaussianLLK(obj.x,obj.muu,obj.prec);
            logposteriorprobability=bsxfun(@plus,logGaussianLikelihood,obj.lpii);
            logposteriorprobability=bsxfun(@minus,logposteriorprobability,...
                max(logposteriorprobability,[],1));
            postprob=exp(logposteriorprobability);
            postprob=bsxfun(@rdivide,postprob,sum(postprob,1));
            obj.z=sample_vector(postprob);
        end
        function sampleWeights(obj)
            
            ngam=zeros(obj.maxClusters,1);
            for n=1:obj.maxClusters
                ngam(n)=sum(obj.z==n);
            end
            obj.ngam=ngam;
            if obj.focused
                [obj.pii,obj.alphaHigherLevel]=HGPsampleweights(ngam,obj.alphaHigherLevel,obj.alpha);
                obj.lpii=log(obj.pii);
            else
                [obj.pii,obj.lpii]=drchrnd(obj.alpha,obj.z,obj.maxClusters);
            end
        end
        function sampleAtoms(obj)
            %% update means and precisions:
            for c=1:obj.maxClusters
                [obj.muu(:,c),obj.prec{c}]=updateNormalWishart(obj.x(:,obj.z==c),...
                    obj.muu0,obj.lam0,obj.sig0,obj.v0);
            end
        end
        function wfalign(obj,smoothing)
            if nargin<2
                smoothing=5;
            end
            
            [x,waveforms]=aligntofeatures(obj.waveforms',smoothing,obj.K);
            obj.x=x;
            obj.waveforms=waveforms';
        end
        function drawPCA(obj,fignum,flag)
            if nargin<2
                fignum=1;
            end
            if nargin<3
                flag=false;
            end
            if flag
                colors=lines(max(obj.z));
                figure(fignum);clf;hold on
                for c=1:max(obj.z)
                    plot(obj.x(1,obj.z==c),obj.x(2,obj.z==c),'.','Color',colors(c,:));
                end
                title(sprintf('Iteration %d',obj.iter))
                hold off
                drawnow
            else
                figure(fignum);clf;hold on
                for c=1:obj.maxClusters
                    plot(obj.x(1,obj.z==c),obj.x(2,obj.z==c),'.','Color',obj.colors(c,:));
                end
                title(sprintf('Iteration %d',obj.iter))
                hold off
                drawnow
            end
        end
        function drawPCA3d(obj,fignum,flag)
            if nargin<2
                fignum=1;
            end
            if nargin<3
                flag=false;
            end
            if flag
                colors=lines(max(obj.z));
                figure(fignum);clf;hold on
                for c=1:max(obj.z)
                    plot3(obj.x(1,obj.z==c),obj.x(2,obj.z==c),obj.x(3,obj.z==c),'.','Color',colors(c,:));
                end
                title(sprintf('Iteration %d',obj.iter))
                hold off
                drawnow
            else
                figure(fignum);clf;hold on
                for c=1:obj.maxClusters
                    plot3(obj.x(1,obj.z==c),obj.x(2,obj.z==c),obj.x(3,obj.z==c),'.','Color',colors(c,:));
                end
                title(sprintf('Iteration %d',obj.iter))
                hold off
                drawnow
            end
        end
        function drawClusters(obj,fignum)

            if nargin<2
                fignum=1;
            end
            figure(fignum);clf;hold on
            colors=lines(max(obj.z));
            for c=1:max(obj.z)
                nz=sum(obj.z==c);
                if nz<=.01*size(obj.x,2)
                    continue
                end
                tmp=(obj.waveforms(:,obj.z==c));
                m=mean(tmp,2);
                s=std(tmp');
                plot(m,'-','Color',colors(c,:),'LineWidth',2);
                ndx=3:3:size(obj.waveforms,1);
                errorbar(ndx,m(ndx),s(ndx),'.','Color',colors(c,:),'LineWidth',2);
            end
            hold off
            drawnow
        end
    end
end
%% Support Functions
function [pii,lpii]=drchrnd(alph,z,numClusters);
% function [pii,lpii]=drchrnd(alph,z,numClusters);
% alph is the hyperparameter (assuming a symmetric distribution)
% z is a vector of assignment variables of length N (one per data point)
% numCluster is the number of clusters
% pii is the dirichlet distribution draw, normalized
% lpii is the log of pii, unnormalized (more numerically stable)
N=numel(z);
counts=sparse(z,ones(N,1),ones(N,1),numClusters,1);
pii=gamrnd(alph+counts,1);
lpii=log(pii);
pii=pii./sum(pii);
end
function [lprob]=GaussianLLK(x,muu,prec)
% x is P by N
% muu is P by numClus
% prec is a cell of precision matricies of size numClusters, with each cell
% of size P by P
% returns a numCluster by N matrix of relative loglikelihood values
N=size(x,2);
numClusters=numel(prec);
lprob=zeros(numClusters,N);
for c=1:numClusters
    lprob(c,:)=NLLK(x,muu(:,c),prec{c});
end
end
function [lprob]=NLLK(x,muu,prec)
% x is P by N
% muu is P by 1
% prec is P by P
xm=bsxfun(@minus,x,muu);
term1=sum(log(diag(chol(prec))));
term2=-.5*dot(xm,prec*xm);
lprob=term1+term2;
end
function z=sample_vector(prob)
[K,N]=size(prob);
cumprob=cumsum(prob);
t=rand(1,N);
z=1+sum(bsxfun(@lt,cumprob(1:K-1,:),t));
end
function [muu,prec]=updateNormalWishart(x,mu0,lam0,sig0,v0)
% function [muu,prec]=updateNormalWishart(x,mu0,lam0,sig0,v0)
[P,N]=size(x);
if N==0;
    prec=wishrnd(sig0,v0);
    muu=1./sqrt(lam0)*chol(prec)\randn(P,1)+mu0;
else
    xbar=mean(x,2);
    xm=bsxfun(@minus,x,xbar);
    sigpost=sig0+xm*xm'+lam0*N./(lam0+N)*(xbar-mu0)*(xbar-mu0)';
    vpost=v0+N;
    prec=wishrnd(inv(sigpost),vpost);
    muu=(N*xbar+lam0*mu0)./(lam0+N)+1./sqrt(lam0+N)*(chol(prec)\randn(P,1));
end
end
function llk=GMMloglikelihood(s,z,muu,lamb,mixture_weights);
[N,P]=size(s);
maxClus=numel(mixture_weights);
logLikelihood=zeros(1,N);
for clus=1:maxClus
    ndx=find(z==clus);
    if numel(ndx)==0
        continue
    end
    lambchol=chol(lamb{clus});
    logLikelihood(1,ndx)=...
        sum(diag(lambchol))-.5*sum((lambchol*bsxfun(@minus,s(ndx,:),muu(clus,:))').^2);
end
nz=sparse(z,ones(N,1),ones(N,1),maxClus,1);
llk=sum(logLikelihood)+nz'*mixture_weights;
end
function [weights,alpha,llk]=HGPsampleweights(nz,alpha,alpha_0)
% Sample from a HGP according to:
% M. Zhou and L. Carin, "Augment-and-conquer Negative Binomial Processes,"
% NIPS 2012
%
% dec 8/12/13
minval=1e-5;
c=1;
p=.5;
Ntables=CRTrnd(nz,alpha);
alpha=gamrnd(alpha_0+sum(Ntables,2),1./(c-log(1-p)));
alpha=max(alpha,minval);
weights=gamrnd(alpha+nz,p);
weights=max(weights,minval);
llk=sum(alpha*log(p))-sum(gammaln(alpha))+(alpha-1)'*log(weights)...
    -p/(1-p)*sum(weights)+(alpha_0-1)*sum(log(alpha))-sum(alpha);

end
function t=CRTrnd(m,r)
% Samples the number of tables used in a chinese restaurant process with
% total number of customers m and parameter r.
t=zeros(size(m));
for n=1:numel(m)
    p=r(n)./(r(n):m(n)-1+r(n));
    t(n)=sum(rand(m(n),1)<p(:));
end
end
function [y,newwaves]=aligntofeatures(x,smoothing,K)
% function [y,xaligned]=aligntofeatures(x,smoothing,K)
%
% This function takes a set of waveforms and aligns them on the maximum
% positive slope.  It returns a set of features (y) and aligned waveforms
% (xaligned) where the missing values have been imputed.
%
% The main input is x, which is size N by P, where N is the number of data
% samples and P is the dimensionality.
%
% smoothing is a nonnegative integer that decides how much to smooth the
% input x before alignment.  Defaults to 0 (no smoothing)
%
% K is the feature dimensionality to extract via PPCA.  Defaults to 3.
%
%


%%
if nargin<2;smoothing=0;end
if nargin<3;K=3;end


%%
grad=x(:,2:end)-x(:,1:end-1);
% grad=x;
grad=filter(ones(smoothing,1),1,grad')';
[~,ndxgrad]=max(grad,[],2);
% hist(ndxgrad)
medgndx=round(median(ndxgrad));

%
medgndx=round(median(ndxgrad));
xgshift=zeros(size(x));
[N,P]=size(x);
for n=1:N
    shift=ndxgrad(n)-medgndx;
    usex=max(1,shift+1):min(P,P+shift);
    useshift=max(1,-shift+1):min(P,P-shift);
    xgshift(n,useshift)=x(n,usex);
end
%%
xgshift(xgshift==0)=nan;
q=sum(isnan(xgshift),2);
[pc,W,data_mean,xr,evals,percentVar]=verbeekppca(xgshift,K+2);
newwaves=xgshift;
Q=isnan(xgshift);
newwaves(Q)=xr(Q);
y=W(1:K,:);
y=bsxfun(@minus,y,mean(y,2));
y=bsxfun(@rdivide,y,std(y')');
1;
end
function [pc,W,data_mean,xr,evals,percentVar]=verbeekppca(data,k)
% PCA applicable to
%   - extreme high-dimensional data (e.g., gene expression data) and
%   - incomplete data (missing data)
%
% probabilistic PCA (PPCA) [Verbeek 2002]
% based on sensible principal components analysis [S. Roweis 1997]
%  code slightly adapted by M.Scholz
%
% pc = ppca(data)
% [pc,W,data_mean,xr,evals,percentVar]=ppca(data,k)
%
%  data - inclomplete data set, d x n - matrix
%          rows:    d variables (genes or metabolites)
%          columns: n samples
%
%  k  - number of principal components (default k=2)
%  pc - principal component scores  (feature space)
%       plot(pc(1,:),pc(2,:),'.')
%  W  - loadings (weights)
%  xr - reconstructed complete data matrix (for k components)
%  evals - eigenvalues
%  percentVar - Variance of each PC in percent
%
%    pc=W*data
%    data_recon = (pinv(W)*pc)+repmat(data_mean,1,size(data,2))
%
% Example:
%    [pc,W,data_mean,xr,evals,percentVar]=ppca(data,2)
%    plot(pc(1,:),pc(2,:),'.');
%    xlabel(['PC 1 (',num2str(round(percentVar(1)*10)/10),'%)',]);
%    ylabel(['PC 2 (',num2str(round(percentVar(2)*10)/10),'%)',]);
%
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin==1
    k=2
end


[C,ss,M,X,Ye]=ppca_mv(data',k,0,0);
xr=Ye';
W=C';
data_mean=M';
pc=X';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate variance of PCs

for i=1:size(data,1)  % total variance, by using all available values
    v(i)=var(data(i,~isnan(data(i,:))));
end
total_variance=sum(v(~isnan(v)));

evals=nan(1,k);
for i=1:k
    data_recon = (pinv(W(i,:))*pc(i,:)); % without mean correction (does not change the variance)
    evals(i)=sum(var(data_recon'));
end

percentVar=evals./total_variance*100;

%    cumsumVarPC=nan(1,k);
%   for i=1:k
%     data_recon = (pinv(W(1:i,:))*pc(1:i,:)) + repmat(data_mean,1,size(data,2));
%     cumsumVarPC(i)=sum(var(data_recon'));
%   end
%   cumsumVarPC
end
function [C, ss, M, X,Ye] = ppca_mv(Ye,d,dia,plo);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% original code by Jakob Verbeek
% implements probabilistic PCA for data with missing values,
% using a factorizing distrib. over hidden states and hidden observations.
%
%  - The entries in Ye that equal NaN are assumed to be missing. -
%
% [C, ss, M, X, Ye ] = ppca_mv(Y,d,dia,plo);
%
% Y   (N by D)  N data vectors
% d   (scalar)  dimension of latent space
% dia (binary)  if 1: printf objective each step
% plo (binary)  if 1: plot first PCA direction each step.
%               if 2: plot eigenimages
%
% ss  (scalar)  isotropic variance outside subspace
% C   (D by d)  C*C' +I*ss is covariance model, C has scaled principal directions as cols.
% M   (D by 1)  data mean
% X   (N by d)  expected states
% Ye  (N by D)  expected complete observations (interesting if some data is missing)
%
% J.J. Verbeek, 2002. http://www.science.uva.nl/~jverbeek
%

%threshold = 1e-3;     % minimal relative change in objective funciton to continue
threshold = 1e-5;

if plo; set(gcf,'Double','on'); end

[N,D] = size(Ye);

Obs   = ~isnan(Ye);
hidden = find(~Obs);
missing = length(hidden);

% compute data mean and center data
if missing
    for i=1:D;  M(i) = mean(Ye(find(Obs(:,i)),i)); end;
else
    M = mean(Ye);
end
Ye = Ye - repmat(M,N,1);

if missing;   Ye(hidden)=0;end

r     = randperm(N);
C     = Ye(r(1:d),:)';     % =======     Initialization    ======
C     = randn(size(C));
CtC   = C'*C;
X     = Ye * C * inv(CtC);
recon = X*C'; recon(hidden) = 0;
ss    = sum(sum((recon-Ye).^2)) / ( (N*D)-missing);

count = 1;
old   = Inf;


while count          %  ============ EM iterations  ==========
    
    if plo; plot_it(Ye,C,ss,plo);    end
    
    Sx = inv( eye(d) + CtC/ss );    % ====== E-step, (co)variances   =====
    ss_old = ss;
    if missing; proj = X*C'; Ye(hidden) = proj(hidden); end
    X = Ye*C*Sx/ss;          % ==== E step: expected values  ====
    
    SumXtX = X'*X;                              % ======= M-step =====
    C      = (Ye'*X)  / (SumXtX + N*Sx );
    CtC    = C'*C;
    ss     = ( sum(sum( (C*X'-Ye').^2 )) + N*sum(sum(CtC.*Sx)) + missing*ss_old ) /(N*D);
    
    objective = N*(D*log(ss) +trace(Sx)-log(det(Sx)) ) +trace(SumXtX) -missing*log(ss_old);
    rel_ch    = abs( 1 - objective / old );
    old       = objective;
    
    count = count + 1;
    if ( rel_ch < threshold) & (count > 5); count = 0;end
    if dia; fprintf('Objective: M %s    relative change: %s \n',objective, rel_ch ); end
    
end             %  ============ EM iterations  ==========


C = orth(C);
[vecs,vals] = eig(cov(Ye*C));
[vals,ord] = sort(diag(vals));
ord = flipud(ord);
vecs = vecs(:,ord);

C = C*vecs;
X = Ye*C;

% add data mean to expected complete data
Ye = Ye + repmat(M,N,1);


% ====  END ===
end


function [zhat,qz,mu,sigma]=DPmergefit(x,Kmax,alph,k0,v0,muu0,phi0,FMM);
% function [zhat,qz,mu,sigma]=DPmergefit(x,Kmax,alph,k0,v0,muu0,phi0);
% x is P by N, where P is the dimensionality and N is the number of samples
% Kmax is the maximum number of clusters, used for initialization
% alph is the concentration parameter in the DP
% [muu0,k0,phi0,v0] are the parameters in the NIW prior for the DP atoms

%% Initialize data:
[P,N]=size(x);
K=Kmax;
[L,C]=kmeansplusplus(x,K);

% learn DP parameters:
maxK=10;
mu=zeros(P,K);
xs=zeros(P,K);
xx=cell(K,1);
nx=zeros(1,K);
for k=1:K
    nx(k)=sum(L==k);
    xs(:,k)=sum(x(:,L==k),2);
    xx{k}=x(:,L==k)*x(:,L==k)';
end
mu(:,1:K)=C;
logprob=zeros(K,N);
lpii=zeros(maxK,1);
piihat=nx./N;
offset=1e-12;
%% VB parameters:
clear Ls Ks
for iter=1:300
    piihat=nx./N;
    ahat=1+nx;
    bhat=sum(nx)-cumsum(nx);
    bhat=bhat+alph;
    tmp=psi(ahat)-psi(ahat+bhat);
    tmp2=psi(bhat)-psi(ahat+bhat);
    lpii=tmp+cumsum(tmp2)-tmp2;
    lpii(end)=log(1-sum(exp(lpii(1:end-1))));
    if sum(isnan(lpii))>0
        1;
    end
    %% Get probs:
    prob=getProbs(x,muu0,phi0,k0,v0,nx,xs,xx,lpii);
    %% Rearrange:
    nx=sum(prob,2);
    [nx,rendx]=sort(nx,'descend');
    prob=prob(rendx,:);
    %% Calculate Sufficient Statistics:
    [nx,xs,xx]=getSS(x,prob);
    %%
    %     muhat=bsxfun(@rdivide,xs,nx(:)');
    %     for k=1:K
    %         varhat(k)=(xx{k}-muhat(:,k)*muhat(:,k)'*nx(k))./nx(k);
    %     end
    
    
    %% Merge step:
    if mod(iter,3)==0
        lprob=log(prob+offset);
        H=diag(-sum(prob.*lprob,2));
        M=zeros(K);Mp=zeros(K);Mnw=zeros(K);
        [s1,~,s2]=stickbreakinglowerbound(nx,alph);
        S=eye(K)*s1;
        Sa=eye(K)*s2;
        for k1=1:K
            for k2=1:K
                if k1==k2
                    %                 M(k1,k2)=-sum(log(diag(chol(xx{k1}+v0*phi0))))+...
                    %                     .5*xs(:,k1)'*xs(:,k1);
                    try
                        [M(k1,k2),Mp(k1,k2),Mnw(k1,k2)]=NIWlowerbound(nx(k1),xs(:,k1),xx{k1},muu0,k0,phi0,v0);
                    catch
                        keyboard
                    end
                else
                    H(k1,k2)=-sum((prob(k1,:)+prob(k2,:)).*log(prob(k1,:)+prob(k2,:)+offset));
                    %                 M(k1,k2)=-sum(log(diag(chol(xx{k1}+xx{k2}+v0*phi0))))+...
                    %                     .5*(xs(:,k1)+xs(:,k2))'*(xs(:,k1)+xs(:,k2));
                    [M(k1,k2),Mp(k1,k2),Mnw(k1,k2)]=NIWlowerbound(nx(k1)+nx(k2),xs(:,k1)+xs(:,k2),xx{k1}+xx{k2},muu0,k0,phi0,v0);
                    nxtmp=nx;nxtmp(k1)=nx(k1)+nx(k2);nxtmp=nxtmp([1:k2-1,k2+1:end]);
                    [S(k1,k2),~,Sa(k1,k2)]=stickbreakinglowerbound(nxtmp,alph);
                end
            end
        end
        O=FMM*(ones(K)-eye(K));
        Mn=M-bsxfun(@plus,diag(M),diag(M)')+diag(diag(M));
        Mpn=Mp-bsxfun(@plus,diag(Mp),diag(Mp)')+diag(diag(Mp));
        Mnwn=Mnw-bsxfun(@plus,diag(Mnw),diag(Mnw)')+diag(diag(Mnw));
        Hn=H-bsxfun(@plus,diag(H),diag(H)')+diag(diag(H));
        Sn=S-bsxfun(@plus,diag(S),diag(S)')/2;
        San=Sa-bsxfun(@plus,diag(Sa),diag(Sa)')/2;
        Q=Mn+Hn+Sn+O;
        %         Q=Mn+H+S+O;
        %         Q2=Q-bsxfun(@plus,diag(Q),diag(Q)')+diag(diag(Q));
        Q2=Q;
        % bsxfun(@rdivide,xs,nx(:)')
        %% Merge
        Q2;
        [k1,k2]=find(max(Q2(:))==Q2,1,'first');
        val=Q2(k1,k2);
        [k1,k2,val];
        if val>0
            nx(k1)=nx(k1)+nx(k2);
            nx=nx([1:k2-1,k2+1:end]);
            xs(:,k1)=xs(:,k1)+xs(:,k2);
            xs=xs(:,[1:k2-1,k2+1:end]);
            xx{k1}=xx{k1}+xx{k2};
            xx=xx([1:k2-1,k2+1:end]);
            prob(k1,:)=prob(k1,:)+prob(k2,:);
            prob=[prob(1:k2-1,:);prob(k2+1:end,:)];
            K=K-1;
            fprintf('merged,K=%d\n',K);
        end
    end
    %     pause
    %% Calculate lower bound
    [Lsb,~,Lalloc]=stickbreakinglowerbound(nx,alph);
    Lallocs(iter)=Lalloc;
    Ln=zeros(K,1);
    Lp=zeros(K,1);
    Lnw=zeros(K,1);
    for k=1:K
        [Ln(k),Lp(k),Lnw(k)]=NIWlowerbound(nx(k),xs(:,k),xx{k},muu0,k0,phi0,v0);
    end
    lprob=log(prob+offset);
    Lent=sum(-sum(prob.*lprob,2));
    Lps(iter)=sum(Lp);
    Lnws(iter)=sum(Lnw);
    L=Lsb+sum(Ln)+Lent+K*FMM;
    Lsbs(iter)=Lsb;
    Lns(iter)=sum(Ln);
    Lents(iter)=Lent;
    Ls(iter)=L;
    
    Ks(iter)=K;
    fprintf('Current Lower Bound: %s\n',L)
    if iter>40
        reldiff=(Ls(iter)-Ls(iter-1))./(Ls(5)-Ls(4));
        rds(iter)=reldiff;
        
        if mod(iter,3)~=0
            if rds(iter)<1e-3
                break
            end
        end
        
    end
end
%%
Lq=Ls-Lps;
qz=prob;
[~,zhat]=max(qz,[],1);
mu=bsxfun(@rdivide,xs,nx'+k0);
sigma=cell(K,1);
for k=1:K
    sigma{k}=(xx{k}+mu(:,k)*mu(:,k)'*nx(k)-xs(:,k)*mu(:,k)'-mu(:,k)*xs(:,k)'+phi0)./(v0+nx(k));
end
end

function r=multigammaln(a,p)
% function r=multigammaln(a,p)
% Evaluates ln(\Gamma_p(a))
al=a+(1-(1:p))/2;
r=p*(p-1)/4*log(pi)+sum(gammaln(al));
1;
end

function ElncholX=InvWishartElogchol(Phi,n);

p=size(Phi,1);
q=psi(.5*(n+1-(1:p)));
% try
s=svd(Phi);
% catch
%     keyboard
% end
% if min(s)<1e-30*max(s)
%     ElnCholX=-1e100;
% else
ElncholX=-sum(q)-p*log(2)+sum(log(s));
end

function ElncholX=WishartElogchol(V,n);

p=size(V,1);
q=psi(.5*(n+1-1:p));
% try
% s=svd(V);
% catch
keyboard
% end
% if min(s)<1e-30*max(s)
%     ElncholX=-1e100;
% else
ElncholX=sum(q)+p*log(2)+sum(log(s));
end

function [L,Elnx,Eln1x]=Betalowerbound(alphahat,betahat,alpha0,beta0);
t=psi(alphahat+betahat);
Elnx=psi(alphahat);
Eln1x=psi(betahat);
L=(alpha0-alphahat)*Elnx+(beta0-betahat)*Eln1x-(alpha0+beta0-alphahat-betahat)*t+...
    betaln(alphahat,betahat)-betaln(alpha0,beta0);
Elnx=Elnx-t;
Eln1x=Eln1x-t;
if isnan(L)
    1;
end
end
function prob=getProbs(x,muu0,phi0,k0,v0,nx,xs,xx,lpii);
[P,K]=size(xs);
for k=1:K
    m=(xs(:,k)+muu0)./(nx(k)+k0);
    v=(xx{k}+m*m'*nx(k)-xs(:,k)*m'-m*xs(:,k)'+phi0)./(v0+nx(k));
    cv=chol(v);%sum(log(diag(cv)))
    ElogSigma=InvWishartElogchol(v*(v0+nx(k)),v0+nx(k));
    logprob(k,:)=lpii(k)-.5*ElogSigma-.5*sum((cv'\bsxfun(@minus,x,m)).^2,1);
end
logprob=bsxfun(@minus,logprob,max(logprob,[],1));
prob=exp(logprob);
prob=bsxfun(@rdivide,prob,sum(prob,1));
if sum(isnan(prob(:)))>0
    1;
end
end
function [nx,xs,xx]=getSS(x,prob)
P=size(x,1);
[K,N]=size(prob);
nx=sum(prob,2);
xs=zeros(P,K);
xx=cell(K,1);
for k=1:K
    xw=bsxfun(@times,prob(k,:),x);
    xs(:,k)=sum(xw,2);
    xx{k}=xw*x';
end
if sum(isnan(xs(:)))>0
    1;
end
end
function [L,Lp,Lnw]=NIWlowerbound(nk,sx,sxx,mu0,lambda0,Phi0,nu0)
% [L]=NIWlowerbound(nk,sx,sxx,mu0,lambda0,Phi0,nu0)
% constants ignored
p=numel(mu0);
Emu=(lambda0*mu0+sx)./(lambda0+nk);
% try
ESigma=(Phi0+sxx+nk*Emu*Emu'-sx*Emu'-Emu*sx')./(nu0+nk-p-1);
if sum(isnan(ESigma(:)))>0
    1;
end
ElogSigma=InvWishartElogchol(ESigma*(nu0+nk-p-1),nu0+nk);
% catch
%     keyboard
% end
EPrec=inv(ESigma)*(nk+nu0)./(nu0+nk-p-1);
%%
Lx=-nk/2*ElogSigma+(EPrec*Emu)'*sx-.5*nk*Emu'*EPrec*Emu-.5*sum(dot(EPrec,sxx))-nk./(nk+lambda0)*(nk+nu0)./(nk+nu0-p-1);
Ln3=p/2*log(lambda0./(lambda0+nk))-.5*lambda0*Emu'*EPrec*Emu+.5*(nk./(nk+lambda0))*p*((nk+nu0-p-1)/(nk+nu0));
Lh2=.5*nk*p*log(2)+nu0/2*log(det(Phi0))+(nk+nu0)*sum(log(diag(chol(1./(nu0+nk)*EPrec))))...
    -multigammaln(nu0/2,p)+multigammaln((nu0+nk)/2,p)...
    +nk/2*ElogSigma+.5*trace(EPrec*(sxx+nk*Emu*Emu'-sx*Emu'-Emu*sx'));
L=Lx+Ln3+Lh2;
Lp=Lx;
Lnw=Ln3+Lh2;
end
function [Lsb,Lbeta,Lalloc]=stickbreakinglowerbound(nx,sb_parameter);
alpha0=1;
beta0=sb_parameter;
alphahat=alpha0+nx;
betahat=sum(nx)-cumsum(nx);
betahat=betahat+beta0;
Lbeta=zeros(numel(betahat),1);
Elnb=zeros(numel(betahat),1);
Eln1b=zeros(numel(betahat),1);
for k=1:numel(Lbeta);
    [Lbeta(k),Elnb(k),Eln1b(k)]=Betalowerbound(alphahat(k),betahat(k),alpha0,beta0);
end
Elw=Elnb+cumsum(Eln1b)-Eln1b;
Elw(end)=log(1-sum(exp(Elw(1:end-1))));
Lalloc=nx'*Elw;
Lsb=sum(Lbeta)+Lalloc;
1;
end
function [L,C] = kmeansplusplus(X,k)
%KMEANS Cluster multivariate data using the k-means++ algorithm.
%   [L,C] = kmeans(X,k) produces a 1-by-size(X,2) vector L with one class
%   label per column in X and a size(X,1)-by-k matrix C containing the
%   centers corresponding to each class.

%   Version: 2013-02-08
%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%
%   References:
%   [1] J. B. MacQueen, "Some Methods for Classification and Analysis of
%       MultiVariate Observations", in Proc. of the fifth Berkeley
%       Symposium on Mathematical Statistics and Probability, L. M. L. Cam
%       and J. Neyman, eds., vol. 1, UC Press, 1967, pp. 281-297.
%   [2] D. Arthur and S. Vassilvitskii, "k-means++: The Advantages of
%       Careful Seeding", Technical Report 2006-13, Stanford InfoLab, 2006.

L = [];
L1 = 0;

while length(unique(L)) ~= k
    
    % The k-means++ initialization.
    C = X(:,1+round(rand*(size(X,2)-1)));
    L = ones(1,size(X,2));
    for i = 2:k
        D = X-C(:,L);
        D = cumsum(sqrt(dot(D,D,1)));
        if D(end) == 0, C(:,i:k) = X(:,ones(1,k-i+1)); return; end
        C(:,i) = X(:,find(rand < D/D(end),1));
        [~,L] = max(bsxfun(@minus,2*real(C'*X),dot(C,C,1).'));
    end
    
    % The k-means algorithm.
    while any(L ~= L1)
        L1 = L;
        for i = 1:k, l = L==i; C(:,i) = sum(X(:,l),2)/sum(l); end
        [~,L] = max(bsxfun(@minus,2*real(C'*X),dot(C,C,1).'),[],1);
    end
    
end
end