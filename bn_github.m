%% clear workspace and load data
clear all; close all; clc;
opts = delimitedTextImportOptions("NumVariables", 7);
opts.DataLines = [4, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["Var1", "Disaster", "BeginDate", "Var4", "CPIAdjustedCost", "Var6", "Var7"];
opts.SelectedVariableNames = ["Disaster", "BeginDate", "CPIAdjustedCost"];
opts.VariableTypes = ["string", "categorical", "double", "string", "double", "string", "string"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, ["Var1", "Var4", "Var6", "Var7"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Disaster", "Var4", "Var6", "Var7"], "EmptyFieldRule", "auto");
tbl = readtable("/Users/bbcael/Documents/work/dogs/billion/events-US-1980-2025.csv", opts); % replace with file location
c = tbl.Disaster; % type
y = tbl.BeginDate; % year
d = tbl.CPIAdjustedCost; % damages
clear opts tbl
C = zeros(size(c)); C(c=="Severe Storm") = 1; % binary classifier for severe storms
y = floor(y./10000); y = y-min(y); % years since start of record
d = d./1000; % damages in billions

%% figure 1 and naive correlation
close all;
scatter(1980+y(C==1),d(C==1),100,[.5 .05 .5],'filled','markerfacealpha',0.75);
hold on;
set(gca,'yscale','log','ticklabelinterpreter','latex','fontsize',18,'yticklabel',{"1","10","100"})
box on;
axis([1979.1 2024.9 1 225])
ylabel('Damage [Billion 2024 USD]','interpreter','latex')
r = corr(y,d)
clear ans lgnd r;
yyaxis right
ylabel('Damage [US GDP Basis Points]','interpreter','latex')
ylim([.3427 77.1])
set(gca,'yscale','log','ytick',[1 2 5 10 20 50])
yyaxis left
scatter(1980+y(C==0),d(C==0),100,[.7 .35 .1],'filled','markerfacealpha',0.75);
lgnd = legend('Severe Storms','Other Disasters');
set(lgnd,'interpreter','latex','fontsize',16,'location','northwest')
exportgraphics(gcf,'Fig1.pdf','ContentType','vector')

%% show severe storms and other disasters are increasing at different rates
Y = unique(y); 
for i = 1:length(Y); 
    D0(i) = sum(y==Y(i) & C==0); 
    D1(i) = sum(y==Y(i) & C==1);
    D(i) = sum(y==Y(i)); 
end
mdl = fitglm(Y,D0,'linear','Distribution','poisson') % non-SS frequency increasing
mdl = fitglm(Y,D1,'linear','Distribution','poisson') % SS also increasing, faster
% seen another way -- probability that a given event is a SS is increasing
mdl = fitglm(y,C,'Distribution','binomial') % SS are becoming relatively more prevalent

%% figures 2a,b -- visualize increases in frequency

figure
subplot(222)
scatter(Y+1980,D0,100,[.7 .35 .1],'filled');
box on;
hold on;
set(gca,'fontsize',16,'ticklabelinterpreter','latex');
ylabel('Other Events per Year','interpreter','latex');
axis([1979.5 2024.5 0 11])
plot(Y+1980,exp(.85+.027.*Y),'k','linewidth',3);
title('Increasing Frequency of Other Disasters','interpreter','latex')

subplot(221)
scatter(Y+1980,D1,100,[.5 .05 .5],'filled');
box on;
hold on;
set(gca,'fontsize',16,'ticklabelinterpreter','latex');
ylabel('Severe Storms per Year','interpreter','latex');
axis([1979.5 2024.5 0 20])
plot(Y+1980,exp(-.71+.079.*Y),'k','linewidth',3);
title('Increasing Frequency of Severe Storms','interpreter','latex')
clear mdl Y D0 D1 D i;

%% show that storms are less costly & non-storms can be grouped, including droughts
[phat,pci] = mle(d-1,'distribution','gp')
[phat,pci] = mle(d(c~="Severe Storm")-1,'distribution','gp') % significantly different when removing severe storms
[phat,pci] = mle(d(c=="Severe Storm")-1,'distribution','gp') % severe storms significantly different than all and other disasters
[phat,pci] = mle(d(c=="Tropical Cyclone")-1,'distribution','gp') % no other group different than non-SS distribution
[phat,pci] = mle(d(c=="Wildfire")-1,'distribution','gp') %
[phat,pci] = mle(d(c=="Flooding")-1,'distribution','gp') %
[phat,pci] = mle(d(c=="Freeze")-1,'distribution','gp') % too few to compute
[phat,pci] = mle(d(c=="Winter Storm")-1,'distribution','gp') %
sum(c=="Drought") % but too few droughts to really take this into consideration, esp. as prevalence isn't changing
mdl = fitglm(y(c=="Drought"),y(c=="Drought"),'linear','Distribution','poisson') % drought frequency almost identical to 'other other' disasters
[phat,pci] = mle(d(c~="Severe Storm" & c~="Drought")-1,'distribution','gp') % not different when removing droughts
[phat,pci] = mle(d(c~="Severe Storm" & c~="Tropical Cyclone")-1,'distribution','gp') % not different when removing droughts
[phat,pci] = mle(d(c~="Severe Storm" & c~="Flooding")-1,'distribution','gp') % not different when removing droughts
[phat,pci] = mle(d(c~="Severe Storm" & c~="Winter Storm")-1,'distribution','gp') % not different when removing droughts
[phat,pci] = mle(d(c~="Severe Storm" & c~="Wildfire")-1,'distribution','gp') % not different when removing droughts

clear ans phat pci mdl;
%% show GP is a good fit – figures 2c,d

subplot(224)
[phat,pci] = mle(d(c~="Severe Storm")-1,'distribution','gp')
[Y,X] = ecdf(sort(d(C==0))); X = X(1:end-1); 
ec = Y; Y = Y(1:end-1);
Y = gpcdf(sort(d(C==0)),phat(1),phat(2),1); tc = Y;
[~,pval] = kstest2(ec(1:end-1),tc)
Y = gpinv(Y,phat(1)-.06,phat(2)+.06,1);
err = sum(abs(X-Y))./length(d(C==0));
plot(1:max([X; Y]),1:max([X; Y]),'k','linewidth',2)
hold on;
scatter(X,Y,50,[.7 .35 .1],'filled');
set(gca,'xscale','log','yscale','log')
box on;
axis([1 Inf 1 Inf])
set(gca,'ticklabelinterpreter','latex','fontsize',16)
ylabel('Empirical Quantiles [\$Bn]','interpreter','latex')
xlabel('GPD Quantiles [\$Bn]','interpreter','latex')
clear ans pci phat X Y err ec tc pval;
title('Damage Distribution of Other Disasters','interpreter','latex')

subplot(223)
[phat,pci] = mle(d(c=="Severe Storm")-1,'distribution','gp')
Y = gpcdf(sort(d(C==1)),phat(1),phat(2),1); tc = Y;
Y = gpinv(Y,phat(1),phat(2),1);
[ec,X] = ecdf(sort(d(C==1))); %X = X(1:end-1);
[~,pval] = kstest2(ec(1:end-1),tc)
err = sum(abs(X-Y))./length(d(C==0));
plot(1:(1+max([X; Y])),1:(1+max([X; Y])),'k','linewidth',2)
hold on;
scatter(X,Y,50,[.5 .05 .5],'filled');
set(gca,'xscale','log','yscale','log')
box on;
axis([1 15 1 15])
set(gca,'ticklabelinterpreter','latex','fontsize',16)
ylabel('Empirical Quantiles [\$Bn]','interpreter','latex')
xlabel('GPD Quantiles [\$Bn]','interpreter','latex')
clear ans pci phat X Y err ec tc pval;
title('Damage Distribution of Severe Storms','interpreter','latex')

exportgraphics(gcf,'Fig2.pdf','ContentType','vector')
%% check other disaster mixture isn't changing
D = d(c~="Severe Storm"); Y = y(c~="Severe Storm"); q = c(c~="Severe Storm");
Q = unique(q);
for i = 1:length(Q);
    b = zeros(size(q));
    b(q==Q(i)) = 1;
    mdl = fitglm(Y,b,'Distribution','binomial');
    t(i) = table2array(mdl.Coefficients(2,3));
end
clear mdl b i q Q;
t = t./norminv(.025/6) % bonferroni correction for non-storm composition
% only one that is even borderline is freezes, which only have 9 events
clear D Y t;

%% evaluate whether tails are increasing via quantile regression
for i = 20:80;
    %[p,stats]=quantreg(y(C==0),d(C==0),i./100,1,1000); 
    [p,stats]=quantreg(y(c=="Wildfire"),d(c=="Wildfire"),i./100,1,1000); 
    P0(i) = sum(stats.pboot(:,1)>0)./1000;
    %[p,stats]=quantreg(y(C==1),d(C==1),i./100,1,1000); 
    %[p,stats]=quantreg(y(c=="Winter Storm"),d(c=="Winter Storm"),i./100,1,1000); 
    %P1(i) = sum(stats.pboot(:,1)>0)./1000;
    clear p stats;
    i
end
clear i;
%plot(1:99,P0./200,1:99,P1./200) 
    % upper part of SS distribution is increasing
    % non-SS distribution is becoming more spread
%sum(P1(3:end-2)>195 | P1(3:end-2)<5)./(95*.05) %  negligible evidence of nonstationarity for severe storms
    % found in 6 %iles, expected in 5 at 5% level
%sum(P0(3:end-2)>195 | P0(3:end-2)<5)./(95*.05) %  not so for non-storms
%clear P0 P1;

%%
%pctls = 8:92; P0 = P0(8:92);
pctls = 1:99; P0 = P0(1:99);
scatter(pctls(P0<.1 | P0>.9),P0(P0<.1 | P0>.9),100,'k','filled');
hold on;
scatter(pctls(P0>.1 & P0<.9),P0(P0>.1 & P0<.9),100,'k');
plot(0:100,0.1+0.*(0:100),'k');
plot(0:100,0.9+0.*(0:100),'k');
box on;
axis([0 100 0 1])
xlabel('Percentile','interpreter','latex')
ylabel('Probability of Increase','interpreter','latex')
set(gca,'ticklabelinterpreter','latex','fontsize',16)

%%
%pctls = 11:89;  P1 = P1(11:89);
pctls = 1:99; P1 = P1(1:99);
figure;
scatter(pctls(P1<.1 | P1>.9),P1(P1<.1 | P1>.9),100,'k','filled');
hold on;
scatter(pctls(P1>.1 & P1<.9),P1(P1>.1 & P1<.9),100,'k');
plot(0:100,0.1+0.*(0:100),'k');
plot(0:100,0.9+0.*(0:100),'k');
box on;
axis([0 100 0 1])
xlabel('Percentile','interpreter','latex')
ylabel('Probability of Increase','interpreter','latex')
set(gca,'ticklabelinterpreter','latex','fontsize',16)

%% check for other individual components of non-SS to decompose changes
for i = 1:99;
    [p,stats]=quantreg(y(c=="Wildfire"),d(c=="Wildfire"),i./100,1,200); % not for wildfires
    %[p,stats]=quantreg(y(c=="Drought"),d(c=="Drought"),i./100,1,200); % not for drought
    %[p,stats]=quantreg(y(c=="Winter Storm"),d(c=="Winter Storm"),i./100,1,200); % not for winter storms
    %[p,stats]=quantreg(y(c=="Freeze"),d(c=="Freeze"),i./100,1,200); % not for freezes
    P(i) = sum(stats.pboot(:,1)>0);
    clear p stats;
    i
end
clear i;
q = 5./sum(c=="Wildfire")
%q = 5./sum(c=="Drought")
%q = 5./sum(c=="Winter Storm")
%q = 5./sum(c=="Freeze")
plot(round(100*q):(100-round(100*q)),P(round(100*q):(100-round(100*q))))
axis([0 100 0 200])
clear P ans q;


%% evidence of nonstationarity

phat = mle(d(C==1)-1,'distribution','gp');
[phat,pci] = mle(d(C==1)-1,'pdf',@(x,a,b,c,d)gppdf(x,a+d.*y(C==1),abs(c+b.*y(C==1))),'Start',[phat(1) 0 phat(2) 0])
pval = 1-normcdf((phat(2)./(phat(2)-pci(1,2))).*(3.96/2)) % SS scale parameter increasing, Eq. 2: 97.5% chance
phat = mle(d(C==1)-1,'distribution','gp');
[phat,pci] = mle(d(C==1)-1,'pdf',@(x,a,b,c)gppdf(x,a,abs(c+b.*y(C==1))),'Start',[phat(1) 0 phat(2)])
pval = 1-normcdf((phat(2)./(phat(2)-pci(1,2))).*(3.96/2)) % SS scale parameter increasing, Eq. 4: 93.2% chance
phat = mle(d(C==0)-1,'distribution','gp');
[phat,pci] = mle(d(C==0)-1,'pdf',@(x,a,b,c,d)gppdf(x,a+d.*y(C==0),abs(c+b.*y(C==0))),'Start',[phat(1) 0 phat(2) 0])
pval = 1-normcdf((phat(4)./(phat(4)-pci(1,4))).*(3.96/2)) % OD shape parameter increasing, Eq. 2: 98.4% chance
phat = mle(d(C==0)-1,'distribution','gp');
[phat,pci] = mle(d(C==0)-1,'pdf',@(x,a,b,c)gppdf(x,a+b.*y(C==0),c),'Start',[phat(1) 0 phat(2)])
pval = 1-normcdf((phat(2)./(phat(2)-pci(1,2))).*(3.96/2)) % OD shape parameter increasing, Eq. 3: 97.4% chance

%% NPGP model

%c(120) = []; C(120) = []; d(120) = []; y(120) = []; % to exclude katrina in historical contextual case

Y = unique(y); 
for i = 1:length(Y); 
    D0(i) = sum(y==Y(i) & C==0); 
    D1(i) = sum(y==Y(i) & C==1); 
end
mdl = fitglm(Y,D0,'linear','Distribution','poisson'); % rate for non-SS
r0 = table2array(mdl.Coefficients(:,1:2));
mdl = fitglm(Y,D1,'linear','Distribution','poisson'); % rate for SS
r1 = table2array(mdl.Coefficients(:,1:2));
%mdl = fitglm(zeros(size(Y)),D0,'linear','Distribution','poisson'); % for constant-frequency case
%r0 = table2array(mdl.Coefficients(:,1:2));
%mdl = fitglm(zeros(size(Y)),D1,'linear','Distribution','poisson');
%r1 = table2array(mdl.Coefficients(:,1:2));
clear mdl Y D0 D1;
[phat,pci] = mle(d(C==0)-1,'distribution','gp'); % use for constant magnitude case
%p0(1:2,1) = phat; p0(1:2,2) = (pci(2,1:2)-pci(1,1:2))./3.96; % constant magnitude case
[phat,pci] = mle(d(C==0)-1,'pdf',@(x,a,c,b)gppdf(x,a+b.*y(C==0),c),'Start',[phat(1) phat(2) 0])
p0(1:3,1) = phat; p0(1:3,2) = (pci(2,1:3)-pci(1,1:3))./3.96;
clear phat pci;
[phat,pci] = mle(d(C==1)-1,'distribution','gp'); % use for constant magnitude case
%p1(1:2,1) = phat; p1(1:2,2) = (pci(2,1:2)-pci(1,1:2))./3.96; % constant magnitude case
[phat,pci] = mle(d(C==1)-1,'pdf',@(x,a,c,b)gppdf(x,a,abs(c+b.*y(C==1))),'Start',[phat(1) phat(2) 0])
p1(1:3,1) = phat; p1(1:3,2) = (pci(2,1:3)-pci(1,1:3))./3.96;
clear phat pci C d y ans c;
%% for uncertainty-free version
%p0(:,2) = 0; p1(:,2) = 0; r0(:,2); r1(:,2) = 0;
%%
%y = 46:50; % for near-future case
%y = 45; % for present case
y = 0:44; % for historical case
nboot = 10000;
for i = 1:nboot;
    b_p0i = p0(1,1)+randn(1).*p0(1,2);
    b_p0ii = p0(2,1)+randn(1).*p0(2,2);
    b_p0iii = p0(3,1)+randn(1).*p0(3,2); % comment out for constant magntiude case
    b_p1i = p1(1,1)+randn(1).*p1(1,2);
    b_p1ii = p1(2,1)+randn(1).*p1(2,2);
    b_p1iii = p1(3,1)+randn(1).*p1(3,2); % comment out for constant magntiude case
    b_r0i = r0(1,1)+randn(1).*r0(1,2);
    b_r0ii = r0(2,1)+randn(1).*r0(2,2);
    b_r1i = r1(1,1)+randn(1).*r1(1,2);
    b_r1ii = r1(2,1)+randn(1).*r1(2,2);
    N = []; S = [];
    for j = 1:length(y);
        b_n0 = poissrnd(exp(b_r0i+y(j).*b_r0ii));
        b_n1 = poissrnd(exp(b_r1i+y(j).*b_r1ii));
        %b_n0 = poissrnd(exp(b_r0i+44.*b_r0ii)); % for no extrapolation case
        %b_n1 = poissrnd(exp(b_r1i+44.*b_r1ii)); % for no extrapolation case
        b_d0 = gprnd(b_p0i+y(j).*b_p0iii,b_p0ii,1,1,b_n0);
        b_d1 = gprnd(b_p1i,b_p1ii+b_p1iii.*y(j),1,1,b_n1);
        %b_d0 = gprnd(b_p0i,b_p0ii,1,1,b_n0); % constant magnitude case
        %b_d1 = gprnd(b_p1i,b_p1ii,1,1,b_n1); % constant magnitude case
        %b_d0 = gprnd(b_p0i+44.*b_p0iii,b_p0ii,1,1,b_n0); % for no extrapolation case
        %b_d1 = gprnd(b_p1i,b_p1ii+b_p1iii.*44,1,1,b_n1);       
        N(end+1:end+b_n0) = b_d0;
        S(end+1:end+b_n1) = b_d1;
    end
    i
    ha_max(i) = max([N S]);
    N(N>250) = 202; % throw out events >katrina
    ha_sum_alt(i) = sum([N S]);
    hs_sum(i) = sum(S);
    clear b_*% N S;
end
clear i j ans nboot p0 p1 r0 r1 y;

clearvars -EXCEPT ha_max ha_sum_alt hs_sum;
save model_v4_hist.mat;

%% figure 3

clear all; close all; clc; load model_v4_fut_baseline.mat; A = ha_sum_alt; 
load model_v4_fut_constfreq.mat; R = ha_sum_alt; 
load model_v4_fut_constmag.mat; S = ha_sum_alt;
load model_v4_fut_certain.mat; K = ha_sum_alt;
clearvars -EXCEPT A R S K;
[y,x] = ksdensity(log10(S));
figure
subplot(1,5,1:3)
p2 = plot(x,y,':','linewidth',2.5,'color',[.5 .05 .5])
hold on;
[y,x] = ksdensity(log10(R));
p3 = plot(x,y,'--','linewidth',2.5,'color',[.6 .25 .25])
[y,x] = ksdensity(log10(K));
p4 = plot(x,y,'-.','linewidth',2.5,'color',[.5 .5 .05])
[y,x] = ksdensity(log10(A));
p1 = plot(x,y,'linewidth',5,'color',[.05 .5 .5])
axis([1.9 3.6 0 2.65])
lgnd = legend([p1 p2 p3 p4],'Baseline','Stationary Magnitudes','Stationary Frequencies','Certain Parameters')
set(lgnd,'interpreter','latex','fontsize',16,'location','northwest')
set(gca,'ticklabelinterpreter','latex','fontsize',16,'ytick',[])
set(gca,'xtick',[2 2.3 2.6 3 3.3 3.6],'xticklabel',{'100','200','400','1000','2000','4000'})
xlabel('2026-2030 Damages [Billion 2024 USD]','interpreter','latex')
ylabel('Probability Density','interpreter','latex')

subplot(1,5,4:5)
load model_v4_hist.mat; H = ha_sum_alt
[y,x] = ksdensity(log10(H));
plot(x,y,'linewidth',3,'color',[.05 .5 .5])
hold on;
plot(linspace(3.47,3.47),linspace(0,2.23),'--k','linewidth',3)
axis([2.9 4.1 0 2.3])
set(gca,'ticklabelinterpreter','latex','fontsize',16,'ytick',[],'xtick',[3 3.3 3.6 4],'xticklabel',{'1','2','4','10'})
xlabel('1980-2024 Damages [\$Tn]','interpreter','latex')
ylabel('Probability Density','interpreter','latex')

exportgraphics(gcf,'Fig3.pdf','ContentType','vector')

%% attribution exercise -- repeat above but normalize damages by GDP growth

clear all; close all; clc;
opts = delimitedTextImportOptions("NumVariables", 7);
opts.DataLines = [4, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["Var1", "Disaster", "BeginDate", "Var4", "CPIAdjustedCost", "Var6", "Var7"];
opts.SelectedVariableNames = ["Disaster", "BeginDate", "CPIAdjustedCost"];
opts.VariableTypes = ["string", "categorical", "double", "string", "double", "string", "string"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, ["Var1", "Var4", "Var6", "Var7"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Disaster", "Var4", "Var6", "Var7"], "EmptyFieldRule", "auto");
tbl = readtable("/Users/bbcael/Documents/work/dogs/billion/events-US-1980-2025.csv", opts);
c = tbl.Disaster;
y = tbl.BeginDate;
d = tbl.CPIAdjustedCost;
clear opts tbl
C = zeros(size(c)); C(c=="Severe Storm") = 1;
y = floor(y./10000); y = y-min(y);
d = d./1000;

opts = delimitedTextImportOptions("NumVariables", 2);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["Var1", "gdp"];
opts.SelectedVariableNames = "gdp";
opts.VariableTypes = ["string", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var1", "EmptyFieldRule", "auto");
tbl = readtable("/Users/bbcael/Documents/work/dogs/billion/GDPC1.csv", opts);
gdp = tbl.gdp;
clear opts tbl

g = zeros(45,1);
for i = 1:length(g);
    g(i) = sum(gdp((4.*(i-1)+1):(4.*i)))./4;
end
clear i gdp;
g = g./min(g);

%g = 2.*g;

dn = d;
for i = 1:45;
    dn(y==(i-1)) = dn(y==(i-1))./g(i);
end
clear i;

yn = y; cn = c; Cn = C; 
yn = yn(dn>1); cn = cn(dn>1); Cn = Cn(dn>1); dn = dn(dn>1);

Y = unique(yn); 
for i = 1:length(Y); 
    D0(i) = sum(yn==Y(i) & Cn==0); 
    D1(i) = sum(yn==Y(i) & Cn==1); 
    D(i) = sum(yn==Y(i)); 
end
mdl = fitglm(Y,D0,'linear','Distribution','poisson')
mdl = fitglm(Y,D1,'linear','Distribution','poisson')

phat = mle(dn(Cn==0)-1,'distribution','gp');
[phat,pci] = mle(dn(Cn==0)-1,'pdf',@(x,a,b,c)gppdf(x,a+b.*yn(Cn==0),c),'Start',[phat(1) 0 phat(2)]) % almost...
phat = mle(dn(Cn==0)-1,'distribution','gp');
[phat,pci] = mle(dn(Cn==0)-1,'pdf',@(x,a,b,c)gppdf(x,a,abs(c+b.*yn(Cn==0))),'Start',[phat(1) 0 phat(2)])
phat = mle(dn(Cn==1)-1,'distribution','gp');
[phat,pci] = mle(dn(Cn==1)-1,'pdf',@(x,a,b,c)gppdf(x,a+b.*yn(Cn==1),c),'Start',[phat(1) 0 phat(2)])
phat = mle(dn(Cn==1)-1,'distribution','gp');
[phat,pci] = mle(dn(Cn==1)-1,'pdf',@(x,a,b,c)gppdf(x,a,abs(c+b.*yn(Cn==1))),'Start',[phat(1) 0 phat(2)])