%% Version with 3rd category

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

%%
c3 = 'Tropical Cyclone';
Y = unique(y); 
for i = 1:length(Y); 
    D0(i) = sum(y==Y(i) & C==0 & c~=c3); 
    D1(i) = sum(y==Y(i) & C==1 & c~=c3); 
    D2(i) = sum(y==Y(i) & c==c3);
end
mdl = fitglm(Y,D0,'linear','Distribution','poisson'); % rate for non-SS
r0 = table2array(mdl.Coefficients(:,1:2));
mdl = fitglm(Y,D1,'linear','Distribution','poisson'); % rate for SS
r1 = table2array(mdl.Coefficients(:,1:2));
mdl = fitglm(Y,D2,'linear','Distribution','poisson'); % rate for SS
r2 = table2array(mdl.Coefficients(:,1:2));
clear mdl Y D0 D1 D2;
C(c==c3) = 2;
[phat,pci] = mle(d(C==0)-1,'distribution','gp');
[phat,pci] = mle(d(C==0)-1,'pdf',@(x,a,c,b)gppdf(x,a+b.*y(C==0),c),'Start',[phat(1) phat(2) 0])
p0(1:3,1) = phat; p0(1:3,2) = (pci(2,1:3)-pci(1,1:3))./3.96;
clear phat pci;
[phat,pci] = mle(d(C==1)-1,'distribution','gp');
[phat,pci] = mle(d(C==1)-1,'pdf',@(x,a,c,b)gppdf(x,a,abs(c+b.*y(C==1))),'Start',[phat(1) phat(2) 0])
p1(1:3,1) = phat; p1(1:3,2) = (pci(2,1:3)-pci(1,1:3))./3.96;
clear phat pci;
[phat,pci] = mle(d(C==2)-1,'distribution','gp');
[phat,pci] = mle(d(C==2)-1,'pdf',@(x,a,c,b)gppdf(x,a,abs(c+b.*y(C==2))),'Start',[phat(1) phat(2) 0])
p2(1:3,1) = phat; p2(1:3,2) = (pci(2,1:3)-pci(1,1:3))./3.96;
clear phat pci C d y ans c;

y = 45:49;
nboot = 10000;
for i = 1:nboot;
    b_p0i = p0(1,1)+randn(1).*p0(1,2);
    b_p0ii = p0(2,1)+randn(1).*p0(2,2);
    b_p0iii = p0(3,1)+randn(1).*p0(3,2);
    b_p1i = p1(1,1)+randn(1).*p1(1,2);
    b_p1ii = p1(2,1)+randn(1).*p1(2,2);
    b_p1iii = p1(3,1)+randn(1).*p1(3,2);
    b_p2i = p2(1,1)+randn(1).*p2(1,2);
    b_p2ii = p2(2,1)+randn(1).*p2(2,2);
    b_p2iii = p2(3,1)+randn(1).*p2(3,2);
    b_r0i = r0(1,1)+randn(1).*r0(1,2);
    b_r0ii = r0(2,1)+randn(1).*r0(2,2);
    b_r1i = r1(1,1)+randn(1).*r1(1,2);
    b_r1ii = r1(2,1)+randn(1).*r1(2,2);
    b_r2i = r2(1,1)+randn(1).*r2(1,2);
    b_r2ii = r2(2,1)+randn(1).*r2(2,2);
    N = []; S = []; T = [];
    for j = 1:length(y);
        b_n0 = poissrnd(exp(b_r0i+y(j).*b_r0ii));
        b_n1 = poissrnd(exp(b_r1i+y(j).*b_r1ii));
        b_n2 = poissrnd(exp(b_r2i+y(j).*b_r2ii));
        b_d0 = gprnd(b_p0i+y(j).*b_p0iii,b_p0ii,1,1,b_n0);
        b_d1 = gprnd(b_p1i,b_p1ii+b_p1iii.*y(j),1,1,b_n1);
        b_d2 = gprnd(b_p2i,b_p2ii+b_p2iii.*y(j),1,1,b_n2);
        N(end+1:end+b_n0) = b_d0;
        S(end+1:end+b_n1) = b_d1;
        T(end+1:end+b_n2) = b_d2;
    end
    i
    N(N>202) = 202;
    T(T>202) = 202;
    ha_sum_alt(i) = sum([N S T]);
    clear b_*;
end
clear i j ans nboot p0 p1 p2 r0 r1 r2 y;
clearvars -EXCEPT ha_sum_alt;
% then compare to e.g. median or other %ile damages from baseline 2-category case 

%% locations of severe storms

clear all; close all; clc;
opts = delimitedTextImportOptions("NumVariables", 7);
opts.DataLines = [4, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["name", "c", "t", "Var4", "d", "Var6", "Var7"];
opts.SelectedVariableNames = ["name", "c", "t", "d"];
opts.VariableTypes = ["string", "categorical", "double", "string", "double", "string", "string"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, ["name", "Var4", "Var6", "Var7"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["name", "c", "Var4", "Var6", "Var7"], "EmptyFieldRule", "auto");
tbl = readtable("/Users/bbcael/Documents/work/dogs/billion/events-US-1980-2025.csv", opts);
name = tbl.name;
c = tbl.c;
t = tbl.t;
d = tbl.d;
d = d./1000;
clear opts tbl;

d = d(c=='Severe Storm');
name = name(c=='Severe Storm');
t = t(c=='Severe Storm');
clear c;
for i = 1:length(name);
    v = split(name(i));
    w(1:length(v),i) = v;
    clear v;
end

kansas = sum(count(w,'Kansas'));
minnesota = sum(count(w,'Minnesota'));
oklahoma = sum(count(w,'Oklahoma'));
mississippi = sum(count(w,'Mississippi'));
center = sum(count(w,'Center'));
wv = sum(count(w,'West Virginia'));
central = sum(count(w,'Central'));
midwest = sum(count(w,'Midwest'));
plains = sum(count(w,'Plains'));
ohio = sum(count(w,'Ohio'));
pennsylvania = sum(count(w,'Pennsylvania'));
northeast = sum(count(w,'Northeast'));
midatlantic = sum(count(w,'Mid-Atlantic'));
east = sum(count(w,'East'));

slow = central + midwest + plains + ohio + kansas + minnesota + east + oklahoma + mississippi + center + wv + pennsylvania + northeast + midatlantic;
slow(slow>1) = 1;

rockies = sum(count(w,'Rockies'));
colorado = sum(count(w,'Colorado'));
arizona = sum(count(w,'Arizona'));
tennessee = sum(count(w,'Tennessee'));
texas = sum(count(w,'Texas'));
west = sum(count(w,'West'))-sum(count(w,'West Virginia'));
northwest = sum(count(w,'Northwest'));
southwest = sum(count(w,'Southwest'));
southeast = sum(count(w,'Southeast'));
southern = sum(count(w,'Southern'));
mountain = sum(count(w,'Mountain'));
western = sum(count(w,'Western'));

fast = rockies + colorado + arizona + tennessee + texas + northwest + southwest + southeast + southern + mountain + western;
fast(fast>1) = 1;

t = floor(t./10000);
i = find(fast+slow==1); 
t = t(i)-1980; fast = fast(i); slow = slow(i);
mdl = fitglm(t,slow','Distribution','binomial');

pval = table2array(mdl.Coefficients(2,4))

clearvars -EXCEPT pval;
