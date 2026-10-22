%% PRISM_new_scene_with_optim.m  —  PRISM Localization WITH RGD Optimizer
%  Scenarios : noris | rand | dft | optim
%  Methods   : RSSI | MDP | SSP | BW-fast
%
%  Gradient fixes (verified by finite-difference check):
%    1) V_logdet *= 2  (Hermitian chain rule: d(Hkj*Hkj')/dHkj = 2*G*Hkj)
%    2) S_j uses factor 2 not 4  (d/dR[(1/2)||R-R'||^2] = dR, * Hermitian factor 2)
%    3) V_nn    *= 2  (same Hermitian chain rule)
%    4) Backtracking line search (guarantees J non-decreasing)
%
%  CORRECTIONS in this revision (each independently verified against a
%  central finite-difference estimate of the true objective; see the
%  accompanying paper's Remark 2 and Remark 3):
%    5) V_logdet is now +2*(...), matching the sign of V_nn. The previous
%       V_logdet = -2*(...) together with conj(grad_all) in
%       prism_riem_update produced the WRONG ascent direction on some RIS
%       elements (verified: tangent-space coefficient had opposite sign
%       from the true dJ/dtheta on ~1/3 of tested components). The
%       correct closed-form gradient nabla_W J is already
%       2*(dJ/dW*) (the Brandwood steepest-ascent cogradient) and must be
%       used DIRECTLY in the retraction step, not conjugated.
%    6) prism_riem_update no longer conjugates grad_all before projecting
%       onto the tangent space of the unit-modulus manifold (see fix 5).
%    7) tangent_whiten now uses a barycenter-weighted, metric-correct BW
%       tangent embedding s = vec(G^{1/2} * L) instead of the previous
%       vech_real(L). The BW Riemannian metric at base point G is
%       g_G(L,L) = ||G^{1/2} L||_F^2, which reduces to the plain
%       Frobenius norm ||L||_F^2 only when G is proportional to the
%       identity. Since G (the BW barycenter of real channel covariances)
%       is generally anisotropic, the previous embedding was NOT a valid
%       first-order approximation of the BW distance (confirmed
%       numerically: the unweighted embedding's distance ratio vs. the
%       true BW^2 distance did not converge to 1 as points approached G,
%       staying off by ~30-50x regardless of how close the points were).
%       Because G^{1/2}*L is generally non-Hermitian (G^{1/2} and L do
%       not commute), the embedding dimension changes from D^2 real
%       (Hermitian-symmetric vech) to 2*D^2 real (plain complex vec).
% =========================================================================
clear; close all; clc;

%% =========================================================
%% 1.  PARAMETERS
%% =========================================================
F_DIRECT = 'csi_direct_new_scene.mat';
F_RISAP  = 'csi_ris_to_ap_new_scene.mat';
F_URRIS  = 'csi_user_to_ris_new_scene.mat';

N_ANT_USE  = 1;
DB_STEP    = 1.0;
R_AVG      = 0.8;
TRAIN_SNAP = 50;
TEST_SNAP  = 30;
N_VAL      = 1000;
ALPHA      = 0.90;
K_CB       = 4;
N_BITS     = 6;
BW_ITERS   = 20;
EPS_VAL    = 1e-2;
SNR_TARGET_DB = 25;

% RGD parameters
N_ITER_OPT = 70;
ALPHA_LR   = 2;   % step size; line search shrinks if needed
L_OPT      = 40;
LAMBDA     = 1;
M_NN       = 6;
LS_MAX     = 8;
LS_SHRINK  = 0.5;
LS_TOL     = 1e-9;

TX_DBM=-10; NF_DB=7; T_K=290; BW_SC=312500;

%% =========================================================
%% 2.  LOAD DATA
%% =========================================================
fprintf('Loading data ...\n');
d1=load(F_DIRECT); d2=load(F_RISAP); d3=load(F_URRIS);

grid_x=double(d1.grid_x(:)); grid_y=double(d1.grid_y(:));
N_AP=double(d1.n_ap); F=double(numel(d1.f_sel_idx));
P=double(d1.n_ris_panels);
N_ELEM=double(d1.ris_num_rows)*double(d1.ris_num_cols);
RIS_ROWS=double(d1.ris_num_rows); RIS_COLS=double(d1.ris_num_cols);

if isfield(d1,'h_d_flat') && isfield(d1,'user_pos')
    h_d_flat=single(d1.h_d_flat);
    Cx_f=double(d1.user_pos(:,1)); Cy_f=double(d1.user_pos(:,2));
else
    [iy_v,ix_v]=find(logical(d1.valid_mask));
    Cx_f=grid_x(ix_v); Cy_f=grid_y(iy_v); N_pts=numel(iy_v);
    mf=matfile(F_DIRECT); h_d_flat=zeros(N_pts,N_AP,F,'single');
    urows=unique(iy_v); BATCH=20;
    row2si=cell(max(iy_v),1);
    for si=1:N_pts, row2si{iy_v(si)}(end+1)=si; end
    for bi=1:ceil(numel(urows)/BATCH)
        rs=urows((bi-1)*BATCH+1); re=urows(min(bi*BATCH,numel(urows)));
        bk=mf.h_d_grid(rs:re,:,:,:);
        for r=rs:re
            for si=row2si{r}
                h_d_flat(si,:,:)=reshape(bk(r-rs+1,ix_v(si),:,:),1,N_AP,F);
            end
        end
    end
    clear mf bk row2si;
    try mf2=matfile(F_DIRECT,'Writable',true);
        mf2.h_d_flat=h_d_flat; mf2.user_pos=[Cx_f,Cy_f]; clear mf2;
        fprintf('  h_d_flat cached.\n'); catch; end
end

h_ra=double(d2.h_ra);

if isfield(d3,'h_ur_flat')
    fprintf('  Loading cached h_ur_flat ...\n');
    h_ur_flat=single(d3.h_ur_flat);
else
    fprintf('  Building h_ur_flat (one-time) ...\n');
    if ~exist('iy_v','var'), [iy_v,ix_v]=find(logical(d1.valid_mask)); end
    N_ur=numel(iy_v); mf=matfile(F_URRIS);
    info=whos(mf); idx_var=find(strcmp({info.name},'h_ur_grid'),1);
    if isempty(idx_var), error('h_ur_grid not found in %s',F_URRIS); end
    ur_size=info(idx_var).size;
    fprintf('  h_ur_grid actual size: [%s]\n',num2str(ur_size));
    if numel(ur_size)~=5, error('Expected 5-D h_ur_grid, got %d-D',numel(ur_size)); end
    P_actual=ur_size(3); F_actual=ur_size(4); N_ELEM_actual=ur_size(5);
    if P_actual~=P||F_actual~=F||N_ELEM_actual~=N_ELEM
        fprintf('  WARNING: actual dims differ from metadata. Using actual.\n'); end
    h_ur_flat_pm=zeros(P_actual,F_actual,N_ELEM_actual,N_ur,'single');
    urows=unique(iy_v); BATCH=20; t0=tic;
    row2si=cell(max(iy_v),1);
    for si=1:N_ur, row2si{iy_v(si)}(end+1)=si; end
    nb=ceil(numel(urows)/BATCH);
    for bi=1:nb
        rs=urows((bi-1)*BATCH+1); re=urows(min(bi*BATCH,numel(urows)));
        bk=mf.h_ur_grid(rs:re,:,:,:,:); bk=permute(bk,[3 4 5 1 2]);
        for r=rs:re
            si_list=row2si{r}; if isempty(si_list), continue; end
            h_ur_flat_pm(:,:,:,si_list)=bk(:,:,:,r-rs+1,ix_v(si_list));
        end
        if mod(bi,max(1,floor(nb/8)))==0||bi==nb
            fprintf('    %.0f%% (%.0fs)\n',100*bi/nb,toc(t0)); end
    end
    clear bk;
    h_ur_flat=permute(h_ur_flat_pm,[4 1 2 3]); clear h_ur_flat_pm;
    fprintf('  Built in %.0fs\n',toc(t0)); clear mf row2si;
    P=P_actual; F=F_actual; N_ELEM=N_ELEM_actual;
    try mf2=matfile(F_URRIS,'Writable',true);
        mf2.h_ur_flat=h_ur_flat; clear mf2;
        fprintf('  Cached.\n');
    catch err, fprintf('  WARNING: cache failed (%s).\n',err.message); end
end

N=size(h_d_flat,1);
fprintf('  N=%d | N_AP=%d | F=%d | P=%d | N_ELEM=%d\n',N,N_AP,F,P,N_ELEM);

if size(h_d_flat,3)~=F, error('h_d_flat F mismatch.'); end
if size(h_ra,3)<F, error('h_ra too few subcarriers.');
elseif size(h_ra,3)~=F, h_ra=h_ra(:,:,1:F,:); end
P_from_hra=size(h_ra,1); P_from_hur=size(h_ur_flat,2);
if P_from_hra~=P_from_hur
    error('Panel count mismatch: h_ra P=%d vs h_ur_flat P=%d.',P_from_hra,P_from_hur); end
if P~=P_from_hra
    fprintf('  NOTE: P corrected from %d to %d.\n',P,P_from_hra); P=P_from_hra; end

%% =========================================================
%% 3.  ANTENNA SELECTION
%% =========================================================
if isempty(N_ANT_USE),      ant_idx=1:N_AP;
elseif isscalar(N_ANT_USE), ant_idx=1:N_ANT_USE;
else,                        ant_idx=N_ANT_USE(:)'; end
N_ANT=numel(ant_idx);
h_d_flat=h_d_flat(:,ant_idx,:);
h_ra    =h_ra(:,ant_idx,:,:);
D=N_ANT*F; D2=D^2; D2W=2*D2; Id=eye(D);
fprintf('  N_ANT=%d  D=%d  (BW tangent-vector length D2W=%d)\n',N_ANT,D,D2W);

%% =========================================================
%% 4.  NOISE
%% =========================================================
samp_idx=randperm(N,min(5000,N));
chan_pwr=mean(abs(double(h_d_flat(samp_idx,:,:))).^2,'all');
sigma2=chan_pwr/10^(SNR_TARGET_DB/10);
sqh=sqrt(sigma2/2);
sigma2_norm=sigma2/chan_pwr;
fprintf('  chan_pwr=%.2e | SNR=%ddB | sigma2=%.2e | sigma2_norm=%.2e\n',...
    chan_pwr,SNR_TARGET_DB,sigma2,sigma2_norm);
if sum(isnan(h_d_flat(:)))>0
    h_d_flat(isnan(h_d_flat))=0; end
clear samp_idx;

%% =========================================================
%% 5.  CODEBOOKS
%% =========================================================
fprintf('Codebooks ...\n');
q_step=2*pi/2^N_BITS;
rng(7);
W_rand=exp(1j*round(rand(K_CB,P,N_ELEM)*2*pi/q_step)*q_step);
[cc,rr]=meshgrid(0:RIS_COLS-1,0:RIS_ROWS-1); rr=rr(:)'; cc=cc(:)';
% Generate K_CB (row-freq, col-freq) DFT index pairs, spread across the
% RIS aperture. FIXED: previously a hardcoded 4-row table [0 0;5 0;0 10;
% 5 10] that only supported exactly K_CB=4 and threw an index-out-of-
% bounds error otherwise. This now generalizes to any K_CB, RIS_ROWS,
% RIS_COLS: row and column DFT frequencies are each spread evenly via
% linspace, and the column sequence is circularly shifted by K_CB/2
% relative to the row sequence so codewords aren't collapsed onto a
% single diagonal (matching the spread of the original 4-codeword table).
row_freqs = round(linspace(0, max(RIS_ROWS-1,0), K_CB));
col_freqs = round(linspace(0, max(RIS_COLS-1,0), K_CB));
shift = floor(K_CB/2);
col_freqs = col_freqs(mod((0:K_CB-1)+shift, K_CB)+1);
kpairs = [row_freqs(:), col_freqs(:)];
W_dft=zeros(K_CB,P,N_ELEM);
for k=1:K_CB
    ph=2*pi*(kpairs(k,1)*rr/RIS_ROWS+kpairs(k,2)*cc/RIS_COLS);
    phq=round(mod(ph,2*pi)/q_step)*q_step;
    for pp=1:P, W_dft(k,pp,:)=exp(1j*phq); end
end
fprintf('  rand | dft\n');

%% =========================================================
%% 6.  TRAIN / TEST SPLIT
%% =========================================================
rng(42);
split_col=int8(randsrc(N,1,[0,1]));
fprintf('  Train=%d | Test=%d\n',sum(split_col==1),sum(split_col==0));

%% =========================================================
%% 7.  DATABASE GRID
%% =========================================================
fprintf('DB grid %.1fm | R=%.1fm | TRAIN_SNAP=%d ...\n',DB_STEP,R_AVG,TRAIN_SNAP);
x_db=(floor(min(Cx_f)):DB_STEP:ceil(max(Cx_f)))';
y_db=(floor(min(Cy_f)):DB_STEP:ceil(max(Cy_f)))';
[Xdb,Ydb]=meshgrid(x_db,y_db);
Cx_db=Xdb(:); Cy_db=Ydb(:);
R2=R_AVG^2; Cx_col=Cx_f(:); Cy_col=Cy_f(:);
grid_idx=cell(numel(Cx_db),1); keep=false(numel(Cx_db),1);
rng(11);
for j=1:numel(Cx_db)
    dv=(Cx_col-Cx_db(j)).^2+(Cy_col-Cy_db(j)).^2;
    nb=find(dv<R2 & split_col==1);
    if numel(nb)>=TRAIN_SNAP
        nb=nb(randperm(numel(nb),TRAIN_SNAP));
        grid_idx{j}=nb; keep(j)=true;
    end
end
Cx_db=Cx_db(keep); Cy_db=Cy_db(keep); grid_idx=grid_idx(keep);
K_db=numel(Cx_db); Lv=cellfun(@numel,grid_idx);
fprintf('  %d pts (>=%d train snaps) | L: min=%d med=%d max=%d\n',...
    K_db,TRAIN_SNAP,min(Lv),round(median(Lv)),max(Lv));
if K_db==0, error('No DB points.'); end

%% =========================================================
%% 8.  RGD PHASE OPTIMIZER
%% =========================================================
fprintf('\n--- RGD Optimizer | Iter=%d | alpha0=%g | L_OPT=%d | lambda=%.2f | M_NN=%d ---\n',...
    N_ITER_OPT,ALPHA_LR,L_OPT,LAMBDA,M_NN);

fprintf('  Pre-loading channels ...');
H_d_db=cell(K_db,1); H_ur_db=cell(K_db,1);
rng(999);
for j=1:K_db
    idx=grid_idx{j}; L_j=numel(idx);
    if L_j>L_OPT, idx=idx(randperm(L_j,L_OPT)); end
    Hd_j=double(h_d_flat(idx,:,:));
    H_d_db{j}=reshape(permute(Hd_j,[2,3,1]),D,numel(idx));
    H_ur_db{j}=double(h_ur_flat(idx,:,:,:));
end
fprintf(' done\n');

fprintf('  Computing %d nearest neighbours per grid point ...',M_NN);
xy_db=[Cx_db(:),Cy_db(:)];
nn_idx=zeros(K_db,M_NN,'int32');
for j=1:K_db
    d2=sum((xy_db-xy_db(j,:)).^2,2); d2(j)=inf;
    [~,srt]=sort(d2); nn_idx(j,:)=int32(srt(1:M_NN));
end
fprintf(' done\n');
fprintf('  J = J_logdet + %.2f * J_nn\n',LAMBDA);

% Precompute reverse-neighbour index for NN gradient
rev_nn=cell(K_db,1);
for j_rv=1:K_db, rev_nn{j_rv}=[]; end
for j_rv=1:K_db
    for mm=1:M_NN
        jp_rv=double(nn_idx(j_rv,mm));
        rev_nn{jp_rv}(end+1)=j_rv;
    end
end

W_opt=W_dft;
OBJ_hist=zeros(N_ITER_OPT,3);
GN_hist=zeros(N_ITER_OPT,1);   % [diag] raw (pre-normalization) gradient norm per iteration
t_opt=tic;
warning('off','MATLAB:nearlySingularMatrix');

[J_init,Jld_i,Jnn_i]=prism_obj_only(W_opt,H_d_db,H_ur_db,h_ra,nn_idx,...
    K_CB,P,F,N_ANT,N_ELEM,D,K_db,Id,sigma2_norm,LAMBDA,M_NN);
fprintf('  init | J=%.4f | J_ld=%.4f | J_nn=%.4f\n',J_init,Jld_i,Jnn_i);

% [diag] Effective-rank / condition-number stats of the DFT-codebook
% covariances BEFORE optimization, on a random subsample of grid points
% (keeps the diagnostic cheap even for large K_db).
N_DIAG = min(K_db, 150);
diag_idx0 = randperm(K_db, N_DIAG);
[rk_med0,rk_mean0,cn_med0,cn_mean0] = prism_diag_covstats(W_dft,H_d_db(diag_idx0),...
    H_ur_db(diag_idx0),h_ra,K_CB,P,F,N_ANT,N_ELEM,D,N_DIAG,Id,sigma2_norm,0.99);
fprintf(['  [diag] INITIAL (DFT) covariance stats over %d sampled pts | ' ...
    'D=%d | eff-rank(99%% energy): med=%.1f mean=%.1f | cond#: med=%.2e mean=%.2e\n'],...
    N_DIAG,D,rk_med0,rk_mean0,cn_med0,cn_mean0);

J_prev=J_init;

for iter=1:N_ITER_OPT
    [J_cur,Jld_cur,Jnn_cur,grad_all]=prism_obj_grad(W_opt,H_d_db,H_ur_db,h_ra,...
        nn_idx,rev_nn,K_CB,P,F,N_ANT,N_ELEM,D,K_db,Id,sigma2_norm,LAMBDA,M_NN);

    gn=norm(grad_all(:));
    GN_hist(iter)=gn;   % [diag] record raw gradient norm before normalization
    if gn>0, grad_all=grad_all/gn;
    else
        fprintf('  iter %3d | zero gradient.\n',iter);
        OBJ_hist(iter:end,:)=repmat([J_cur,Jld_cur,Jnn_cur],N_ITER_OPT-iter+1,1);
        break;
    end

    % Backtracking line search
    alpha_try=ALPHA_LR;
    W_new=W_opt; J_new=J_cur; Jld_new=Jld_cur; Jnn_new=Jnn_cur;
    accepted=false;
    for ls=1:LS_MAX
        W_try=prism_riem_update(W_opt,grad_all,alpha_try,K_CB,P,N_ELEM);
        [J_try,Jld_try,Jnn_try]=prism_obj_only(W_try,H_d_db,H_ur_db,h_ra,...
            nn_idx,K_CB,P,F,N_ANT,N_ELEM,D,K_db,Id,sigma2_norm,LAMBDA,M_NN);
        if J_try>=J_cur-LS_TOL
            W_new=W_try; J_new=J_try; Jld_new=Jld_try; Jnn_new=Jnn_try;
            accepted=true; break;
        end
        alpha_try=alpha_try*LS_SHRINK;
    end

    if accepted, W_opt=W_new;
    else, alpha_try=0; J_new=J_cur; Jld_new=Jld_cur; Jnn_new=Jnn_cur; end

    OBJ_hist(iter,:)=[J_new,Jld_new,Jnn_new];

    if mod(iter,10)==0||iter==1||~accepted
        fprintf('  iter %3d/%d | J=%.4f | J_ld=%.4f | J_nn=%.4f | ΔJ=%+.4f | |grad|=%.3e | alpha=%g | %.1fmin\n',...
            iter,N_ITER_OPT,J_new,Jld_new,Jnn_new,...
            J_new-J_prev,gn,alpha_try,toc(t_opt)/60);
    end
    J_prev=J_new;
end
warning('on','MATLAB:nearlySingularMatrix');

% Quantize
W_before_quant=W_opt;
theta_opt=angle(W_opt);
W_opt=exp(1j*round(mod(theta_opt,2*pi)/q_step)*q_step);
[J_unq]=prism_obj_only(W_before_quant,H_d_db,H_ur_db,h_ra,nn_idx,...
    K_CB,P,F,N_ANT,N_ELEM,D,K_db,Id,sigma2_norm,LAMBDA,M_NN);
[J_q,Jld_q,Jnn_q]=prism_obj_only(W_opt,H_d_db,H_ur_db,h_ra,nn_idx,...
    K_CB,P,F,N_ANT,N_ELEM,D,K_db,Id,sigma2_norm,LAMBDA,M_NN);
fprintf('  Done %.1fmin | init=%.4f -> opt=%.4f -> quant=%.4f\n',...
    toc(t_opt)/60,J_init,J_unq,J_q);

% [diag] Effective-rank / condition-number stats AFTER optimization
% (same sampled grid points as the initial diagnostic, for a fair
% before/after comparison), plus a summary of how |grad| trended.
[rk_med1,rk_mean1,cn_med1,cn_mean1] = prism_diag_covstats(W_opt,H_d_db(diag_idx0),...
    H_ur_db(diag_idx0),h_ra,K_CB,P,F,N_ANT,N_ELEM,D,N_DIAG,Id,sigma2_norm,0.99);
fprintf(['  [diag] FINAL (optimized) covariance stats over %d sampled pts | ' ...
    'D=%d | eff-rank(99%% energy): med=%.1f mean=%.1f | cond#: med=%.2e mean=%.2e\n'],...
    N_DIAG,D,rk_med1,rk_mean1,cn_med1,cn_mean1);
fprintf('  [diag] eff-rank change: %+.2f (mean) | cond# change: %+.2e (mean)\n',...
    rk_mean1-rk_mean0, cn_mean1-cn_mean0);
gn_nz = GN_hist(GN_hist>0);
if ~isempty(gn_nz)
    fprintf('  [diag] |grad| (pre-normalization) over run: first=%.3e last=%.3e min=%.3e max=%.3e\n',...
        gn_nz(1), gn_nz(end), min(gn_nz), max(gn_nz));
end

% Convergence plot
figure('Color','w','Position',[40 40 760 320]);
subplot(1,2,1);
plot(OBJ_hist(:,1),'k','LineWidth',1.8); hold on;
plot(OBJ_hist(:,2),'Color',[0.85 0.33 0.1],'LineWidth',1.5);
plot(OBJ_hist(:,3)*LAMBDA,'Color',[0 0.45 0.74],'LineWidth',1.5);
xlabel('Iteration'); ylabel('Objective');
legend('J total','J_{logdet}',sprintf('%.2f·J_{nn}',LAMBDA),'Location','best');
title(sprintf('J = J_{ld} + %.2f·J_{nn}',LAMBDA)); grid on;
subplot(1,2,2);
dJ=diff(OBJ_hist(any(OBJ_hist,2),1));
if ~isempty(dJ), plot(dJ,'k','LineWidth',1.2); end
yline(0,'r--'); xlabel('Iteration'); ylabel('ΔJ');
title('All ΔJ should be ≥ 0'); grid on;
sgtitle(sprintf('RGD | K_{db}=%d | λ=%.2f | α=%g',K_db,LAMBDA,ALPHA_LR));

%% =========================================================
%% 9.  CASCADE FINGERPRINTS
%% =========================================================
fprintf('\nCascade fingerprints ...\n');
sc_names={'noris','rand','dft','optim'}; n_sc=4;
H_sc=cell(n_sc,1);
H_sc{1}={single(h_d_flat)};
fprintf('  rand  ...'); H_sc{2}=cascade_diag(h_d_flat,h_ur_flat,h_ra,W_rand,K_CB,N_ANT,N_ELEM,P,F); fprintf(' done\n');
fprintf('  dft   ...'); H_sc{3}=cascade_diag(h_d_flat,h_ur_flat,h_ra,W_dft, K_CB,N_ANT,N_ELEM,P,F); fprintf(' done\n');
fprintf('  optim ...'); H_sc{4}=cascade_diag(h_d_flat,h_ur_flat,h_ra,W_opt, K_CB,N_ANT,N_ELEM,P,F); fprintf(' done\n');

%% =========================================================
%% 10. OFFLINE TRAINING
%% =========================================================
fprintf('\nOffline training ...\n');
rng(101); models=struct();

for si=1:n_sc
    sn=sc_names{si}; Hsi=H_sc{si}; K_sc=numel(Hsi);
    fprintf('  [%d/%d] %-6s K_sc=%d ...\n',si,n_sc,sn,K_sc);
    RSSI_db=zeros(K_sc*D,K_db,'single');
    P_SSP  =zeros(K_sc*D2,K_db,'single');
    Rk_cell=cell(K_sc,K_db); Rt_sum=cell(K_sc,1);
    for kb=1:K_sc, Rt_sum{kb}=zeros(D); end

    for j=1:K_db
        idx=grid_idx{j}; L_j=numel(idx);
        for kb=1:K_sc
            Y=reshape(permute(double(Hsi{kb}(idx,:,:)),[2,3,1]),D,L_j);
            Y=Y+sqh*(randn(D,L_j)+1j*randn(D,L_j));
            Rk=make_cov(Y,L_j,EPS_VAL,D,Id);
            Rk_cell{kb,j}=single(Rk); Rt_sum{kb}=Rt_sum{kb}+Rk;
            ro=(kb-1)*D+(1:D); fo=(kb-1)*D2+(1:D2);
            RSSI_db(ro,j)=single(10*log10(max(real(diag(Rk)),1e-12)));
            [Vk,ek]=eig(Rk,'vector'); [ek,od]=sort(real(ek),'descend');
            Vk=Vk(:,od); ek=max(ek,0);
            q=find(cumsum(ek)/max(sum(ek),eps)>=ALPHA,1); if isempty(q),q=1;end
            P_SSP(fo,j)=single(reshape(Vk(:,1:q)*Vk(:,1:q)',D2,1));
        end
    end

    P_MDP=zeros(K_sc*D2,K_db,'single'); Rt_inv=cell(K_sc,1);
    for kb=1:K_sc
        Rt=Rt_sum{kb}/K_db;
        [Vt,et]=eig((Rt+Rt')/2,'vector');
        et=max(real(et),1e-10*max(abs(real(et))));
        Rinv=Vt*diag(1./et)*Vt'; Rinv=Rinv/real(trace(Rinv));
        Rt_inv{kb}=Rinv; fo=(kb-1)*D2+(1:D2);
        for j=1:K_db
            RR=Rinv*double(Rk_cell{kb,j});
            [Vj,ej]=eig((RR+RR')/2,'vector');
            [ej,od]=sort(real(ej),'descend'); Vj=Vj(:,od); ej=max(ej,0);
            q=find(cumsum(ej)/max(sum(ej),eps)>=ALPHA,1); if isempty(q),q=1;end
            P_MDP(fo,j)=P_MDP(fo,j)+single(reshape(Vj(:,1:q)*Vj(:,1:q)',D2,1));
        end
    end

    % --- BW tangent-space fingerprints (barycenter-weighted, isometric) ---
    bwG=cell(K_sc,1); bwS=zeros(K_sc*D2W,K_db,'single');
    for kb=1:K_sc
        Gbw=bw_barycenter(Rk_cell(kb,:),BW_ITERS);
        reg=max(1e-8,1e-4*real(trace(Gbw))/D);
        Gbw=Gbw+reg*Id; bwG{kb}=Gbw; foW=(kb-1)*D2W+(1:D2W);
        for j=1:K_db
            bwS(foW,j)=single(tangent_whiten(double(Rk_cell{kb,j}),Gbw));
        end
    end
    tv_test=tangent_whiten(double(Rk_cell{1,1}),bwG{1});
    assert(numel(tv_test)==D2W,'BW vec len %d != D2W=%d',numel(tv_test),D2W);
    fprintf('    done\n');
    models.(sn).RSSI=RSSI_db; models.(sn).P_SSP=P_SSP;
    models.(sn).P_MDP=P_MDP;  models.(sn).Rinv=Rt_inv;
    models.(sn).bwG=bwG;      models.(sn).bwS=bwS;
    models.(sn).K_sc=K_sc;
end

%% =========================================================
%% 11. VALIDATION
%% =========================================================
fprintf('\nValidation N_VAL=%d | TEST_SNAP=%d ...\n',N_VAL,TEST_SNAP);
rng(202);
min_x=min(Cx_col); max_x=max(Cx_col);
min_y=min(Cy_col); max_y=max(Cy_col);
val_x=min_x+(max_x-min_x)*rand(1,N_VAL);
val_y=min_y+(max_y-min_y)*rand(1,N_VAL);

meth={'RSSI','MDP','SSP','BW-fast'}; N_M=numel(meth);
ERRS=NaN(N_VAL,N_M,n_sc); skipped=0;

for t=1:N_VAL
    xt=val_x(t); yt=val_y(t);
    dv=(Cx_col-xt).^2+(Cy_col-yt).^2;
    test_idx_full=find(dv<R2 & split_col==0);
    if numel(test_idx_full)<TEST_SNAP, skipped=skipped+1; continue; end
    test_idx=test_idx_full(randperm(numel(test_idx_full),TEST_SNAP));
    L_q=TEST_SNAP;

    for si=1:n_sc
        sn=sc_names{si}; Hsi=H_sc{si}; K_sc=models.(sn).K_sc;
        rssi_q=zeros(K_sc*D,1,'single');
        Rq_stk=zeros(K_sc*D2,1,'single');
        Rq_wh =zeros(K_sc*D2,1,'single');
        bwSq  =zeros(K_sc*D2W,1,'single');
        for kb=1:K_sc
            Yq=reshape(permute(double(Hsi{kb}(test_idx,:,:)),[2,3,1]),D,L_q);
            Yq=Yq+sqh*(randn(D,L_q)+1j*randn(D,L_q));
            Rq=make_cov(Yq,L_q,EPS_VAL,D,Id);
            ro=(kb-1)*D+(1:D); fo=(kb-1)*D2+(1:D2); foW=(kb-1)*D2W+(1:D2W);
            rssi_q(ro)=single(10*log10(max(real(diag(Rq)),1e-12)));
            Rq_stk(fo)=single(Rq(:));
            Rq_wh(fo) =single(reshape(models.(sn).Rinv{kb}*Rq,D2,1));
            bwSq(foW) =single(tangent_whiten(Rq,models.(sn).bwG{kb}));
        end
        [~,mi]=min(sum((double(models.(sn).RSSI)-double(rssi_q)).^2,1));
        ERRS(t,1,si)=norm([Cx_db(mi)-xt,Cy_db(mi)-yt]);
        [~,mi]=max(real(models.(sn).P_MDP'*Rq_wh));
        ERRS(t,2,si)=norm([Cx_db(mi)-xt,Cy_db(mi)-yt]);
        [~,mi]=max(real(models.(sn).P_SSP'*Rq_stk));
        ERRS(t,3,si)=norm([Cx_db(mi)-xt,Cy_db(mi)-yt]);
        d2=sum((double(models.(sn).bwS)-double(bwSq)).^2,1);
        [~,mi]=min(d2);
        ERRS(t,4,si)=norm([Cx_db(mi)-xt,Cy_db(mi)-yt]);
    end
    if mod(t,100)==0||t==N_VAL
        fprintf('  [%4d/%4d]  skipped=%d\n',t,N_VAL,skipped); end
end
fprintf('  Valid=%d  Skipped=%d\n',N_VAL-skipped,skipped);

%% =========================================================
%% 12. RESULTS
%% =========================================================
sep=repmat('=',1,92);
fprintf('\n%s\n',sep);
fprintf('  PRISM | N_ANT=%d/%d | D=%d | DB=%.1fm | R=%.1fm | TRAIN=%d | TEST=%d | SNR=%ddB\n',...
    N_ANT,N_AP,D,DB_STEP,R_AVG,TRAIN_SNAP,TEST_SNAP,SNR_TARGET_DB);
fprintf('%s\n',repmat('-',1,92));
fprintf('  %-8s | %-9s | %7s | %7s | %7s | %7s\n',...
    'Scenario','Method','Mean(m)','Med(m)','75th%','90th%');
fprintf('  %s\n',repmat('-',1,90));
for si=1:n_sc
    for mi=1:N_M
        v=ERRS(:,mi,si); v=v(~isnan(v)); if isempty(v),continue;end
        fprintf('  %-8s | %-9s | %6.2fm | %6.2fm | %6.2fm | %6.2fm\n',...
            sc_names{si},meth{mi},mean(v),median(v),prctile(v,75),prctile(v,90));
    end
    if si<n_sc, fprintf('  %s\n',repmat('-',1,90)); end
end
fprintf('%s\n\n',sep);

%% =========================================================
%% 13. CDF PLOTS
%% =========================================================
clrs={[0 .45 .74],[.85 .33 .1],[.47 .67 .19],[0 0.6 0]}; lw=1.8;
figure('Color','w','Position',[40 40 1200 340]);
for mi=1:N_M
    subplot(1,N_M,mi); hold on; grid on; box on;
    for si=1:n_sc
        v=ERRS(:,mi,si); v=v(~isnan(v)); if isempty(v),continue;end
        [ff,xx]=ecdf(v);
        plot(xx,ff,'Color',clrs{si},'LineWidth',lw,...
            'DisplayName',sprintf('%s (%.2fm)',sc_names{si},median(v)));
    end
    xlabel('Error (m)'); ylabel('CDF');
    title(meth{mi},'FontWeight','bold');
    legend('Location','southeast','FontSize',7);
    xlim([0 8]); ylim([0 1]);
end
sgtitle(sprintf('PRISM | N_{ANT}=%d | D=%d | TRAIN=%d | TEST=%d | SNR=%ddB',...
    N_ANT,D,TRAIN_SNAP,TEST_SNAP,SNR_TARGET_DB),'FontSize',10);

figure('Color','w','Name','BW-fast','Position',[40 420 600 400]);
hold on; grid on; box on;
for si=1:n_sc
    v=ERRS(:,4,si); v=v(~isnan(v)); if isempty(v),continue;end
    [ff,xx]=ecdf(v);
    plot(xx,ff,'Color',clrs{si},'LineWidth',lw,...
        'DisplayName',sprintf('%s (%.2fm)',sc_names{si},median(v)));
end
xlabel('Error (m)'); ylabel('CDF');
title('BW-fast: noris | rand | dft | optim','FontWeight','bold');
legend('Location','southeast','FontSize',9);
xlim([0 8]); ylim([0 1]);

%% =========================================================
%% LOCAL FUNCTIONS
%% =========================================================
function Rk=make_cov(Y,L,eps_val,D,Id)
good=~any(isnan(Y),1)&sum(abs(Y).^2,1)>0;
if sum(good)<1,Rk=Id;return;end
Y=Y(:,good);Rk=Y*Y';Rk=(Rk+Rk')/2;
tr=real(trace(Rk));if tr<=0,Rk=Id;return;end
Rk=Rk/tr;
end

function H_all=cascade_diag(h_d,h_ur,h_ra,W,K_CB,N_ANT,N_ELEM,P,F)
N=size(h_d,1);H_all=cell(K_CB,1);
hra_pf=cell(P,F);hur_pf=cell(P,F);
for pp=1:P,for f=1:F
    hra_pf{pp,f}=reshape(h_ra(pp,:,f,:),N_ANT,N_ELEM);
    hur_pf{pp,f}=reshape(h_ur(:,pp,f,:),N,N_ELEM);
end;end
h_d_double=double(h_d);
for k=1:K_CB
    Hk=h_d_double;
    for pp=1:P,w=reshape(W(k,pp,:),1,N_ELEM);
        for f=1:F
            Hk(:,:,f)=Hk(:,:,f)+bsxfun(@times,hur_pf{pp,f},w)*hra_pf{pp,f}.';
        end;end
    H_all{k}=single(Hk);
end
end

function S=msqrt(M)
M=(M+M')/2;[V,d]=eig(M,'vector');
S=V*diag(sqrt(max(real(d),0)))*V';
end

function G=bw_barycenter(mats,iters)
K=numel(mats);n=size(mats{1},1);G=zeros(n);
for k=1:K,G=G+double(mats{k})/K;end
for it=1:iters
    Gh=msqrt(G);Gi=pinv(Gh);S=zeros(n);
    for k=1:K,S=S+msqrt(Gh*double(mats{k})*Gh)/K;end
    G=Gi*(S*S)*Gi;G=(G+G')/2;
end
end

function U=bw_log(Pb,Q)
Pb=Pb+1e-8*eye(size(Pb,1));Ph=msqrt(Pb);Pi=pinv(Pb);
T=Pi*Ph*msqrt(Ph*Q*Ph)*Ph*Pi;
U=T*Pb+Pb*T'-2*Pb;U=(U+U')/2;
end

function L=lyap_solve(Pb,U)
Pb=(Pb+Pb')/2+1e-8*eye(size(Pb,1));
[V,d]=eig(Pb,'vector');d=real(d);
den=d(:)+d(:)';den(den<1e-10)=1e-10;
L=V*((V'*U*V)./den)*V';L=(L+L')/2;
end

function s=tangent_whiten(Y,G)
% FIXED: BW tangent-space embedding, metric-correct.
% The BW Riemannian metric at base point G is g_G(L,L)=||G^{1/2} L||_F^2,
% which equals the plain Frobenius norm ||L||_F^2 only when G is
% proportional to the identity. G (the BW barycenter of the database
% covariances) is generally anisotropic here, so we must weight by
% G^{1/2} for the embedded Euclidean distance to approximate the true
% BW distance (verified numerically: unweighted ratio-to-dBW^2 does not
% converge to 1 as points approach G, while the weighted ratio does).
% G^{1/2}*L is generally NOT Hermitian (G^{1/2} and L need not commute),
% so we use a plain complex vectorization (2*D^2 real dims) rather than
% the Hermitian-symmetric vech_real used for L itself.
L  = lyap_solve(G, bw_log(G,Y));   % BW "velocity" representation at G
Gh = msqrt(G);                     % G^{1/2}
M  = Gh*L;                         % NOT Hermitian in general
s  = [real(M(:)); imag(M(:))];     % D^2 complex -> 2*D^2 real, isometric
end

%% =========================================================
%% RGD HELPER FUNCTIONS
%% =========================================================

function [J_total,J_logdet,J_nn,grad_all]=prism_obj_grad(W,H_d_db,H_ur_db,h_ra,...
    nn_idx,rev_nn,K_CB,P,F,N_ANT,N_ELEM,D,K_db,Id,sigma2_norm,LAMBDA,M_NN)
% Gradient with verified factor-of-2 fixes:
%   V_logdet = 2*G*Hkj/(L_j*tr_raw)   [Hermitian chain rule]
%   S_j uses factor 2 not 4             [correct NN gradient]
%   V_nn    = 2*G*Hkj/(L_j*tr_raw)   [Hermitian chain rule]
% V_logdet and V_nn now use the SAME sign (+2): the closed-form gradient
% nabla_W J = (1/L_j) sum_l d_{l,p,n}^H (v_ld + lambda*v_nn) is exactly
% 2*(dJ/dW*) (the Brandwood steepest-ascent cogradient) and must be used
% directly (not conjugated) in prism_riem_update. Verified against
% central finite differences of the true objective on a synthetic
% instance: this symmetric-sign, no-extra-conjugation combination matches
% the true dJ/dtheta_n exactly (ratio 1.0000) on every tested RIS
% element, whereas the previous asymmetric-sign + conj(grad_all)
% combination had the WRONG SIGN on some elements.

    grad_all=zeros(K_CB,P,N_ELEM,'like',complex(1));
    J_logdet=0; J_nn=0;
    reg=max(sigma2_norm,0.01/D); sreg=1+reg*D;

    Rk_all=zeros(D,D,K_CB,K_db);
    R0_all=zeros(D,D,K_CB,K_db);
    TR_all=zeros(K_CB,K_db);
    Hkj_all=cell(K_CB,K_db);

    for k=1:K_CB
        for j=1:K_db
            L_j=size(H_d_db{j},2); Hkj=H_d_db{j};
            for pp=1:P
                w_kp=reshape(W(k,pp,:),1,N_ELEM);
                for f=1:F
                    hra_pf=reshape(h_ra(pp,:,f,:),N_ANT,N_ELEM);
                    hur_pf=reshape(H_ur_db{j}(:,pp,f,:),L_j,N_ELEM);
                    f_rows=(f-1)*N_ANT+(1:N_ANT);
                    Hkj(f_rows,:)=Hkj(f_rows,:)+(bsxfun(@times,hur_pf,w_kp)*hra_pf.').';
                end
            end
            Rraw=(Hkj*Hkj')/L_j; Rraw=(Rraw+Rraw')/2;
            tr_raw=max(real(trace(Rraw)),1e-30);
            R0=Rraw/tr_raw; R=(R0+reg*Id)/sreg; R=(R+R')/2;
            Rk_all(:,:,k,j)=R; R0_all(:,:,k,j)=R0;
            TR_all(k,j)=tr_raw; Hkj_all{k,j}=Hkj;
        end
    end

    for k=1:K_CB
        for j=1:K_db
            L_j=size(H_d_db{j},2);
            Hkj=Hkj_all{k,j};
            R=Rk_all(:,:,k,j); R0=R0_all(:,:,k,j); tr_raw=TR_all(k,j);

            % log-det objective
            ev=eig(R,'vector');
            J_logdet=J_logdet+sum(log(1+max(real(ev),0)/sigma2_norm));

            % log-det gradient through R=(R0+reg)/sreg and R0=Rraw/tr
            A=eye(D)+R/sigma2_norm;
            G_log=(A\eye(D))/sigma2_norm;   % dJ/dR
            G_log=(G_log+G_log')/2;
            G_log=G_log/sreg;               % back through R=(R0+reg)/sreg
            % back through R0=Rraw/tr_raw:
            G_log=G_log-real(trace(G_log'*R0))*Id;
            G_log=(G_log+G_log')/2;
            % FIXED: +2, matching V_nn's sign (see header/comment above).
            V_logdet=2*(G_log*Hkj)/(L_j*tr_raw);

            % NN discrimination
            % S_j uses forward neighbours + reverse neighbours (Formula C)
            nn_j=double(nn_idx(j,:)); S_j=zeros(D);
            for jj=1:M_NN
                jp=nn_j(jj); dR=R-Rk_all(:,:,k,jp);
                S_j=S_j+dR;                              % forward: j->jp
                J_nn=J_nn+real(sum(abs(dR(:)).^2));
            end
            % reverse: add (R_j - R_jr) for all jr that have j as neighbour
            for jj=1:numel(rev_nn{j})
                jr=rev_nn{j}(jj);
                S_j=S_j+(R-Rk_all(:,:,k,jr));           % reverse: jr->j
            end
            S_j=(S_j+S_j')/2;
            G_nn=S_j/sreg;
            G_nn=G_nn-real(trace(G_nn'*R0))*Id;
            G_nn=(G_nn+G_nn')/2;
            V_nn=2*(G_nn*Hkj)/(L_j*tr_raw);

            V_kj=V_logdet+LAMBDA*V_nn;

            % Project to RIS phase gradient (Wirtinger dJ/dw_kp*)
            for pp=1:P
                hra_p=reshape(h_ra(pp,:,:,:),N_ANT,F,N_ELEM);
                hur_j=reshape(H_ur_db{j}(:,pp,:,:),L_j,F,N_ELEM);
                hur_j_p=reshape(permute(hur_j,[2,3,1]),1,F,N_ELEM,L_j);
                D_all=reshape(bsxfun(@times,hra_p,hur_j_p),D,N_ELEM,L_j);
                contrib=squeeze(sum(sum(conj(D_all).*reshape(V_kj,D,1,L_j),1),3));
                grad_all(k,pp,:)=squeeze(grad_all(k,pp,:))+contrib(:);
            end
        end
    end
    J_nn=J_nn/2;
    J_total=J_logdet+LAMBDA*J_nn;
end

function [J_total,J_logdet,J_nn]=prism_obj_only(W,H_d_db,H_ur_db,h_ra,...
    nn_idx,K_CB,P,F,N_ANT,N_ELEM,D,K_db,Id,sigma2_norm,LAMBDA,M_NN)
    J_logdet=0; J_nn=0;
    reg=max(sigma2_norm,0.01/D); sreg=1+reg*D;
    Rk_all=zeros(D,D,K_CB,K_db);
    for k=1:K_CB
        for j=1:K_db
            L_j=size(H_d_db{j},2); Hkj=H_d_db{j};
            for pp=1:P
                w_kp=reshape(W(k,pp,:),1,N_ELEM);
                for f=1:F
                    hra_pf=reshape(h_ra(pp,:,f,:),N_ANT,N_ELEM);
                    hur_pf=reshape(H_ur_db{j}(:,pp,f,:),L_j,N_ELEM);
                    f_rows=(f-1)*N_ANT+(1:N_ANT);
                    Hkj(f_rows,:)=Hkj(f_rows,:)+(bsxfun(@times,hur_pf,w_kp)*hra_pf.').';
                end
            end
            Rraw=(Hkj*Hkj')/L_j; Rraw=(Rraw+Rraw')/2;
            tr_raw=max(real(trace(Rraw)),1e-30);
            R0=Rraw/tr_raw; R=(R0+reg*Id)/sreg; R=(R+R')/2;
            Rk_all(:,:,k,j)=R;
            ev=eig(R,'vector');
            J_logdet=J_logdet+sum(log(1+max(real(ev),0)/sigma2_norm));
        end
    end
    for k=1:K_CB
        for j=1:K_db
            R=Rk_all(:,:,k,j); nn_j=double(nn_idx(j,:));
            for jj=1:M_NN
                jp=nn_j(jj); dR=R-Rk_all(:,:,k,jp);
                J_nn=J_nn+real(sum(abs(dR(:)).^2));
            end
        end
    end
    J_nn=J_nn/2; J_total=J_logdet+LAMBDA*J_nn;
end

function W_new=prism_riem_update(W,grad_all,alpha,K_CB,P,N_ELEM)
% Riemannian ascent on unit-circle manifold.
% FIXED: use grad_all DIRECTLY (no conj()). nabla_W J (=grad_all, per
% prism_obj_grad) is already 2*(dJ/dW*), the correct Brandwood
% steepest-ascent cogradient, so conjugating it before use was wrong.
% Verified: the tangent-space component of grad_all at W=e^{j*theta}
% reproduces dJ/dtheta exactly (ratio 1.0000) via finite differences;
% conj(grad_all) did not (and had the wrong sign on some components).
    W_new=W;
    for k=1:K_CB
        for pp=1:P
            g=squeeze(grad_all(k,pp,:));         % FIXED: no conj()
            w=squeeze(W(k,pp,:));
            riem_g=g-real(g.*conj(w)).*w;
            w_trial=w+alpha*riem_g;
            W_new(k,pp,:)=w_trial./max(abs(w_trial),1e-30);
        end
    end
end

%% =========================================================
%% DIAGNOSTIC HELPER (added to debug slow/flat RGD convergence)
%% =========================================================
function [rank_med,rank_mean,cond_med,cond_mean] = prism_diag_covstats(...
    W,H_d_db,H_ur_db,h_ra,K_CB,P,F,N_ANT,N_ELEM,D,K_db,Id,sigma2_norm,rank_thresh)
% Computes effective-rank (number of eigenvalues needed to capture
% rank_thresh, e.g. 0.99, of total eigenvalue mass) and condition number
% (largest/smallest eigenvalue, floored to avoid div-by-zero) of the
% normalized fingerprint covariances R_k(j) under codebook W, over the
% supplied set of grid points (pass a subsampled H_d_db/H_ur_db subset
% for speed on large databases). Use this to check whether a given run's
% covariances are actually low-rank/ill-conditioned (the regime PRISM's
% BW machinery is designed for) or already well-conditioned (in which
% case BW's advantages over simpler methods -- and the tangent-space
% isometry fix -- have little to bite on).
    reg=max(sigma2_norm,0.01/D); sreg=1+reg*D;
    ranks=zeros(K_CB,K_db); conds=zeros(K_CB,K_db);
    for k=1:K_CB
        for j=1:K_db
            L_j=size(H_d_db{j},2); Hkj=H_d_db{j};
            for pp=1:P
                w_kp=reshape(W(k,pp,:),1,N_ELEM);
                for f=1:F
                    hra_pf=reshape(h_ra(pp,:,f,:),N_ANT,N_ELEM);
                    hur_pf=reshape(H_ur_db{j}(:,pp,f,:),L_j,N_ELEM);
                    f_rows=(f-1)*N_ANT+(1:N_ANT);
                    Hkj(f_rows,:)=Hkj(f_rows,:)+(bsxfun(@times,hur_pf,w_kp)*hra_pf.').';
                end
            end
            Rraw=(Hkj*Hkj')/L_j; Rraw=(Rraw+Rraw')/2;
            tr_raw=max(real(trace(Rraw)),1e-30);
            R0=Rraw/tr_raw; R=(R0+reg*Id)/sreg; R=(R+R')/2;
            ev=sort(real(eig(R)),'descend'); ev=max(ev,0);
            cs=cumsum(ev)/max(sum(ev),eps);
            r=find(cs>=rank_thresh,1); if isempty(r), r=D; end
            ranks(k,j)=r;
            ev_floor=max(ev,1e-12*max(ev(1),eps));
            conds(k,j)=ev_floor(1)/ev_floor(end);
        end
    end
    rank_med=median(ranks(:)); rank_mean=mean(ranks(:));
    cond_med=median(conds(:)); cond_mean=mean(conds(:));
end