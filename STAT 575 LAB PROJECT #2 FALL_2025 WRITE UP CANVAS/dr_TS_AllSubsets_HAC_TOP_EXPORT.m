%======================================================================
% dr_TS_AllSubsets_HAC_TOP_EXPORT.m
% Exhaustive (all subsets) time-series subset selection with HAC kernels.
% Robust to non-SPD HAC matrices (safe wrapper + SPD enforcement).
%
% For each subset S ⊆ {1..p}:
%   - Fit OLS (with or without intercept)
%   - Compute residuals
%   - Get HAC coefficient covariance (TR|BT|PZ|TH|QS), safely
%   - Compute AIC, SBC (BIC), and CICOMP
%   - Export Top-N subsets to LaTeX (.tex) and Word (.rtf)
%   - Save full results to CSV
%======================================================================

clear; clc; close all; format short; rng(1);

%% ---------------- User controls ----------------
filename     = 'smsa_data.xls';     % expects y in col 1, X in cols 2:8
hacKernel    = 'QS';                % 'TR'|'BT'|'PZ'|'TH'|'QS'
addIntercept = true;                % include intercept?
useParallel  = false;               % set true if you opened a parpool
topN         = 1;                   % how many top subsets to export (1 = best only)

%% ---------------- Load data -------------------
SMSA = xlsread(filename);
y = SMSA(:,1);
X = SMSA(:,2:8);             % p=7 predictors (adjust if your file differs)
[n,p] = size(X);

% Newey–West style automatic lag (you can override)
hacLag = floor(4*(n/100)^(2/9));

fprintf('Data: n=%d, p=%d | Kernel=%s | Intercept=%d | hacLag=%d\n', ...
        n, p, hacKernel, addIntercept, hacLag);

%% ---------------- Enumerate all subsets ----------------
tot = 2^p;
Sizes    = zeros(tot,1);
IdxStr   = strings(tot,1);
AIC_all  = nan(tot,1);
SBC_all  = nan(tot,1);
CIC_all  = nan(tot,1);
row = 0;

for m = 0:p
    C = nchoosek(1:p, m);  nC = size(C,1);

    if useParallel && m>0 && nC>1
        Sizes_loc   = zeros(nC,1);
        IdxStr_loc  = strings(nC,1);
        AIC_loc     = nan(nC,1);
        SBC_loc     = nan(nC,1);
        CIC_loc     = nan(nC,1);

        parfor j = 1:nC
            Sj = C(j,:);
            [aicj, sbcj, cicj] = score_subset(X(:,Sj), y, addIntercept, hacKernel, hacLag);
            Sizes_loc(j)  = m;
            IdxStr_loc(j) = mat2str(Sj);
            AIC_loc(j)    = aicj;
            SBC_loc(j)    = sbcj;
            CIC_loc(j)    = cicj;
        end

        A = row + (1:nC);
        Sizes(A)   = Sizes_loc;
        IdxStr(A)  = IdxStr_loc;
        AIC_all(A) = AIC_loc;
        SBC_all(A) = SBC_loc;
        CIC_all(A) = CIC_loc;

    else
        for j = 1:nC
            Sj = C(j,:);
            [aicj, sbcj, cicj] = score_subset(X(:,Sj), y, addIntercept, hacKernel, hacLag);
            Sizes(row+j)   = m;
            IdxStr(row+j)  = mat2str(Sj);
            AIC_all(row+j) = aicj;
            SBC_all(row+j) = sbcj;
            CIC_all(row+j) = cicj;
        end
    end

    row = row + nC;
    fprintf('  done size m=%d (subsets=%d)\n', m, nC);
end

T = table(Sizes, IdxStr, AIC_all, SBC_all, CIC_all, ...
    'VariableNames', {'Size','Indices','AIC','SBC','CICOMP'});

% Best per criterion
[~,ia] = sort(T.AIC);    TopAIC = T(ia(1:min(topN,end)),:);
[~,ib] = sort(T.SBC);    TopSBC = T(ib(1:min(topN,end)),:);
[~,ic] = sort(T.CICOMP); TopCIC = T(ic(1:min(topN,end)),:);

fprintf('\n=== Top-%d subsets ===\n', topN);
disp('AIC  :'); disp(TopAIC);
disp('SBC  :'); disp(TopSBC);
disp('CICOMP:'); disp(TopCIC);

%% ---------------- Save CSV + Export Top-N to LaTeX & RTF -----------
stamp = datestr(now,'yyyy-mm-dd_HHMMSS');
stem  = ['ALL_SUBSETS_', hacKernel, '_', stamp];
writetable(T, [stem '.csv']);
fprintf('\nSaved ALL subsets to: %s.csv\n', stem);

% Build small top tables to export
TopAIC_tbl = addvars(TopAIC, repmat("AIC",height(TopAIC),1), 'Before','Size', 'NewVariableNames','Criterion');
TopSBC_tbl = addvars(TopSBC, repmat("SBC",height(TopSBC),1), 'Before','Size', 'NewVariableNames','Criterion');
TopCIC_tbl = addvars(TopCIC, repmat("CICOMP",height(TopCIC),1),'Before','Size','NewVariableNames','Criterion');
TopAll = [TopAIC_tbl; TopSBC_tbl; TopCIC_tbl];

% Write LaTeX
texFile = [stem, '_TOP', num2str(topN), '.tex'];
write_top_to_latex(TopAll, texFile, hacKernel, topN);
fprintf('Saved LaTeX top-%d table: %s\n', topN, texFile);

% Write Word-friendly (RTF)
rtfFile = [stem, '_TOP', num2str(topN), '.rtf'];
write_top_to_rtf(TopAll, rtfFile, hacKernel, topN);
fprintf('Saved Word (RTF) top-%d table: %s\n', topN, rtfFile);

%% ---------------- Quick summary bar ----------------------
figure('Color','w','Position',[120 120 920 380]);
cats = categorical({'AIC-best','SBC-best','CICOMP-best'});
vals = [TopAIC.AIC(1), TopSBC.SBC(1), TopCIC.CICOMP(1)];
bar(cats, vals); grid on;
ylabel('Criterion value'); title(sprintf('Best-by-criterion (HAC=%s)', hacKernel));

% Trajectory: best value vs model size (for each criterion)
sz = 0:p;
bestBySize_AIC = arrayfun(@(m) min(T.AIC(T.Size==m)), sz);
bestBySize_SBC = arrayfun(@(m) min(T.SBC(T.Size==m)), sz);
bestBySize_CIC = arrayfun(@(m) min(T.CICOMP(T.Size==m)), sz);

figure('Color','w','Position',[120 520 920 380]); hold on;
plot(sz, bestBySize_AIC,'-o','LineWidth',1.8,'DisplayName','AIC');
plot(sz, bestBySize_SBC,'-s','LineWidth',1.8,'DisplayName','SBC');
plot(sz, bestBySize_CIC,'-d','LineWidth',1.8,'DisplayName','CICOMP');
xlabel('Subset size (number of predictors)'); ylabel('Best value at size');
title(sprintf('Best-by-size trajectories (HAC=%s)', hacKernel));
grid on; legend('Location','best');

%% ====================== Helpers ============================
function [AICv, SBCv, CICv] = score_subset(Xs, y, addInt, hacKernel, hacLag)
% Fit OLS for subset, compute residuals, HAC coeff-cov (robustly), then AIC/SBC/CICOMP.
T = numel(y);
if isempty(Xs)
    Xd = ones(T,1); k = 1;
else
    if addInt, Xd = [ones(T,1), Xs]; else, Xd = Xs; end
    k = size(Xd,2);
end

% OLS via fitlm (convenient for hac)
M = fitlm(Xs, y, 'Intercept', addInt);
e = M.Residuals.Raw;

% MLE sigma^2 and Gaussian log-likelihood at MLE
sig2 = max((e'*e)/T, eps);
logL = -0.5*( T*log(2*pi*sig2) + T );

AICv = -2*logL + 2*k;
SBCv = -2*logL + k*log(T);

% HAC coeff-cov with robust wrapper (SPD-enforced)
HAC = safe_hac_beta_cov(M, Xd, e, hacKernel, hacLag);
CICv = cicomp_from_cov(HAC, sig2, T, k);
end
%---------------------------------------------------------------------
function CovB = safe_hac_beta_cov(M, Xd, e, kernel, L)
% Try MATLAB's hac() first. If it fails (non-SPD), build manual NW Cov(beta)
% and enforce SPD in either case.

    % 1) Try MATLAB hac()
    try
        args = {'type','HAC','weights',kernel,'display','off'};
        if ~isempty(L) && L>0
            args = [args, {'bandwidth', L+1}];  % NW heuristic: bandwidth = L+1
        end
        CovB = hac(M, args{:});
        CovB = force_spd(CovB);
        return
    catch
        % fall back to manual NW coeff covariance
    end

    % 2) Manual Newey–West coefficient covariance:
    %    Cov(beta) = (X'X)^(-1) * X' * Omega * X * (X'X)^(-1),
    %    with Omega = HAC covariance of residual vector.
    Omega = hac_cov_from_resid(e, L, kernel);              % T x T (SPD forced)
    XtX   = (Xd' * Xd);
    XtX   = force_spd(XtX);                                 % guard for inversion
    P     = XtX \ eye(size(XtX));                           % (X'X)^(-1)
    CovB  = P * (Xd' * Omega * Xd) * P;                     % k x k
    CovB  = force_spd(CovB);
end

function Omega = hac_cov_from_resid(e, L, kernel)
% Toeplitz HAC covariance of residual vector with kernel weights
    T = numel(e); e = e - mean(e);
    L = max(0, round(L));
    g = zeros(L+1,1);
    for ell=0:L
        g(ell+1) = (e(1+ell:end)'*e(1:end-ell))/T;  % auto-cov at lag ell
    end
    w = zeros(L+1,1);
    for ell=0:L
        w(ell+1) = hac_weight(kernel, ell, L);
    end
    gw = g; gw(2:end) = w(2:end).*g(2:end);
    Omega = toeplitz(gw);
    Omega = force_spd(Omega);
end
%---------------------------------------------------------------------
function w = hac_weight(kernel, ell, L)
% Kernel weights; QS included. Domain: ell=0..L.
    if L<=0, w = (ell==0); return; end
    x = ell/(L+1);
    switch upper(kernel)
        case 'BT'   % Bartlett
            w = 1 - x;
        case 'PZ'   % Parzen
            w = (1 - x)^2 * (1 + 2*x);
        case 'TH'   % Tukey–Hanning (cosine taper)
            w = 0.5 * (1 + cos(pi*x));   % smooth in [0,1]
        case 'TR'   % Tukey (triangular taper)
            w = max(0, 1 - x);
        case 'QS'   % Quadratic Spectral
            if ell==0
                w = 1;
            else
                w = (3/(pi^2*x^2))*(sin(pi*x)/(pi*x) - cos(pi*x));
                w = max(w,0);
            end
        otherwise   % default: delta at 0
            w = (ell==0);
    end
end
%---------------------------------------------------------------------
function A = force_spd(A)
% Symmetrize and lift tiny/negative eigenvalues to ensure SPD
    A = (A + A')/2;
    [V,D] = eig(A);
    d = real(diag(D));
    floorv = max(1e-10, 1e-12*max(1,mean(abs(d))));
    d = max(d, floorv);
    A = V*diag(d)*V';
    A = (A + A')/2;
end
%---------------------------------------------------------------------
function val = cicomp_from_cov(CovB, sig2, n, k)
% Covariance Information Complexity (your formulation)
    CovB = (CovB + CovB')/2;
    CovB = force_spd(CovB);
    ev = eig(CovB);
    evbar = mean(ev);
    if ~isfinite(evbar) || evbar<=0
        mu = mean(ev(ev>0)); if ~isfinite(mu) || mu<=0, mu = 1; end
        CovB = 0.9*CovB + 0.1*mu*eye(size(CovB));
        ev = eig(CovB); evbar = mean(ev);
    end
    C1F = (1/(4*(evbar^2))) * sum((ev - evbar).^2);
    % Literal form from your script:
    val = n*log(2*pi) + n*log(max(sig2,eps)) + n + k + 2*log(n)*C1F;
end
%---------------------------------------------------------------------
function write_top_to_latex(Ttop, fname, kernel, topN)
    fid = fopen(fname,'w');
    fprintf(fid, '%% Auto-generated on %s\n', datestr(now));
    fprintf(fid, '\\begin{table}[h!]\\centering\\small\n');
    fprintf(fid, '\\caption{Top-%d subsets by criterion (HAC kernel: %s)}\n', topN, kernel);
    fprintf(fid, '\\label{tab:top-subsets-%s}\n', lower(kernel));
    fprintf(fid, '\\begin{tabular}{l c c l c c}\n\\toprule\n');
    fprintf(fid, 'Criterion & Size & AIC & Indices & SBC & CICOMP \\\\\n\\midrule\n');

    % Explode into three blocks grouped by criterion for readability
    critOrder = {'AIC','SBC','CICOMP'};
    for c = 1:numel(critOrder)
        mask = strcmp(Ttop.Criterion, critOrder{c});
        TT = Ttop(mask,:);
        for i=1:height(TT)
            fprintf(fid, '%s & %d & %.4f & %s & %.4f & %.4f \\\\\n', ...
                TT.Criterion{i}, TT.Size(i), TT.AIC(i), TT.Indices{i}, ...
                TT.SBC(i), TT.CICOMP(i));
        end
        if c < numel(critOrder) && any(mask)
            fprintf(fid, '\\midrule\n');
        end
    end

    fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
    fclose(fid);
end
%---------------------------------------------------------------------
function write_top_to_rtf(Ttop, fname, kernel, topN)
    fid = fopen(fname,'w');
    fprintf(fid,'{\\rtf1\\ansi\\deff0\n');
    fprintf(fid,'{\\b Top-%d subsets by criterion (HAC kernel: %s)\\b0}\\line\n', topN, kernel);
    fprintf(fid,'\\trowd\\cellx1200\\cellx2200\\cellx3600\\cellx7200\\cellx9000\\cellx10800\n');
    fprintf(fid,'\\intbl Criterion\\cell Size\\cell AIC\\cell Indices\\cell SBC\\cell CICOMP\\cell\\row\n');
    % Keep the same grouping order as LaTeX for consistency
    critOrder = {'AIC','SBC','CICOMP'};
    for c = 1:numel(critOrder)
        mask = strcmp(Ttop.Criterion, critOrder{c});
        TT = Ttop(mask,:);
        for i=1:height(TT)
            fprintf(fid,'\\trowd\\cellx1200\\cellx2200\\cellx3600\\cellx7200\\cellx9000\\cellx10800\n');
            fprintf(fid,'\\intbl %s\\cell %d\\cell %.4f\\cell %s\\cell %.4f\\cell %.4f\\cell\\row\n', ...
                TT.Criterion{i}, TT.Size(i), TT.AIC(i), TT.Indices{i}, ...
                TT.SBC(i), TT.CICOMP(i));
        end
    end
    fprintf(fid,'}\n');
    fclose(fid);
end
