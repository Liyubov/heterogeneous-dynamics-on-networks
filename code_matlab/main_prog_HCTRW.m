% Main program to calculate propagator to compare exact and approximate formulas for Ps(x0|x)
% to study heterogeneities on an interval
% PAPER I on HCTRW (Denis, Liubov, 2018)

%%%%%%%%%% Propagram to study HCTRW on networks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign global variable %%
Nsize = 100;%size of graph should be square for comparing with lattices
tauplus = 1;%parameters of exponent distribution for psi(s)_+=1/(1+s tauplus);
tauminus = 10;
tmax = 2000; %change it locally in functions % tmax1 = 500;  %tmax for propagator P(t)% tmax2 = 100;  % tmax3 = 500;
xfinish = 1; % finishing node if we calculate inverse Laplace
xstart = 50; %starting node MUST be less than Nsize
p = 0.5; %asymmetric parameter for Q CIRCULAR matrix 

%% Assign variables for HCTRW on particular graphs
m = 2; %for CIRCULAR graphs degree of each node>0, 
t_a = 1;% time for calculation of Px0x(t)
t_b = 5; %upper logscale bound for t
tsize = 1000; %number of different t 
t = logspace(t_a, t_b, tsize); %[1:tmax]; %; %logarithmic scale for time 50 points %[1:tmax];
t_points = 50; %number of points for logscale%tmaxFPT =10^3; %tmax for FPT(t)
timelog = logspace(1, 6, t_points); %timescale for FPT
slog = logspace(-5,-1, t_points);
interval = [1:t_points];

close all
disp('******************************')
disp('Start of the HCTRW program ')

%% ****************** check the input *******************************)
if xstart >Nsize
    disp('Enter another XSTART')
end

%% calculate propagators and density FPT P(x) and P(t) for distributional HETEROGENEITIES on interval
disp('Calculate densities FPT')
density_first_pass_func(Nsize, tauplus, tauminus, p, xfinish)

disp('finished')