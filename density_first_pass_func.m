% function to the program from April 2018 
% function calculates FPT only for chain graphs


function density_first_pass_func(Nsize, tauplus, tauminus, p, xfinish)
%% local parameters assignment
talbot_param = 64; %M:   Optional, number of terms to sum for each t (64 is a good guess); highly oscillatory functions require higher M, but this can grow unstable; see test_talbot.m for an example of stability.
alpha =1; % IF ANY parameter for heterogeneous distribution of tau+ and tau-
alphahetero=0.5;
Tpower = 4; %power for T_max
T_max=10^Tpower; %maximum T for which we calculate P_x0x(t)%s_min=0.001; %minimal s for which we calculate P_x0x(s) %x_h = 25; %heterogeneous node, is set while calcaulting propagators
n_sum = 10^4; %number of components taken into account in analytical formula %n_sum2=10^5; %number of components in analytic formula for propagator P(s)% n_sum1 = 10^2;
t_points = 50; %number of points for logscale%tmaxFPT =10^3; %tmax for FPT(t)
timelog = logspace(1, 6, t_points); %timescale for FPT
interval = [1:t_points];

%% calculate density of FPT(t) as Laplace transform using formula sP_x0x0(s)
    function dens_x0x_t_cont = dens_contin(Nsize)
    disp('Calculating CONTINUOUS density fpt for x0x(t) at xind');
    dens_x0x_t_cont = zeros(1, t_points); %density fpt =d FPT/dt
    index=1;
    xstart =50; % xfinish = 1; %final for continous is 0!!
    for tind=timelog 
        for n_i = 1:n_sum %loop through all n to calculate 
            alpha_n = pi*(n_i -1/2); %coefficient for u_nx solution
            c_n = sqrt(2/Nsize);
            u_nx =  c_n*Nsize/alpha_n*(1-cos(alpha_n));% c_n*sin(alpha_n/Nsize) ;  %formula for Px0,1(t)
            u_nx0 = c_n*sin(alpha_n*xstart/Nsize) ;
            D = 1/(2*tauplus); %for a=1 AND tauplus = tauminus
            dens_x0x_t_cont(index) = (dens_x0x_t_cont(index) + u_nx *u_nx0 * exp(-D*tind*alpha_n^2/Nsize^2)*(D*alpha_n^2/Nsize^2));%
        end    
        index=index+1;
    end
    end

    function dens_fpt_t = density_fpt(Nsize, TauPlus, TauMinus, xstart) 
        dens_fpt_t =zeros(t_points, 1);
        index=1;
        for time = timelog %logspace(1,6, t_points) 
            time_int = time:(time+1);
            alpha = 1; %parameter for homogeneous or heterogeneous cases
            xfinish =1; %xfinish is absorbing x
            Propag_dens_fun_t = talbot_inversion(@(svalue) (svalue*propag_s_x_start_x_fin_abs(svalue, xstart, xfinish, Nsize, p, alpha, TauPlus, TauMinus) ),time_int,64); % density for homogeneous case % propag_s_x_start_x_fin_abs(0, x0_ind, xfinish, Nsize, p, alpha, tauplus, tauminus)
            dens_fpt_t(index) = Propag_dens_fun_t(2);            
            index=index+1;
        end    
    end


    function dens_fpt_t = density_fpt_hetero(Nsize, TauPlus, TauMinus, param_hetero, xstart, amplitude_i) % a_tau = 0.1; 
        xfinish =1; %xfinish = absorbing x %        xstart = 100; % tau_period = 0; %parameter of symmetry: if tau_period=1 tau(x)_+ = tau(x)_-=a*x    
        dens_fpt_t =zeros(t_points, 1);
        index=1;
        amplitude =0.5; %default amplitude of sin
        if param_hetero==1          
            for time =timelog %logspace(1,6, t_points) 
            time_int = time:(time+1);
            alpha = 1; %parameter for homogeneous or heterogeneous cases
            a_tau = 2/((Nsize+1)); %normalised coefficient for tau(x)=a_tau*x
            Propag_dens_fun_t = talbot_inversion(@(svalue) svalue*propag_s_x_interval_disoder(svalue, xstart, xfinish, Nsize, p, alpha, TauPlus, TauPlus, param_hetero, amplitude), time_int,talbot_param); 
            dens_fpt_t(index) = Propag_dens_fun_t(2);            
            index=index+1;
            end    
        elseif param_hetero==0 %then we get tau(x) = sinx
            index=1;
            for time =timelog%logspace(1,6, t_points) 
            time_int = time:(time+1);
            alpha = 1; %parameter for homogeneous or heterogeneous cases
            Propag_dens_fun_t = talbot_inversion(@(svalue) svalue*propag_s_x_interval_disoder(svalue, xstart, xfinish, Nsize, p, alpha, TauPlus, TauPlus, param_hetero,amplitude_i), time_int,talbot_param); 
            dens_fpt_t(index) = Propag_dens_fun_t(2);            
            index=index+1;
            end             
        else %then we get tau(x) = 1/x
            index=1;
            for time =timelog%logspace(1,6, t_points) 
            time_int = time:(time+1);
            alpha = 1; %parameter for homogeneous or heterogeneous cases
            %tau_period = 0; %parameter of symmetry: if tau_period=1 tau(x)_+ = tau(x)_-=a*x    %            a_tau = 2/((Nsize+1));
            Propag_dens_fun_t = talbot_inversion(@(svalue) svalue*propag_s_x_interval_disoder(svalue, xstart, xfinish, Nsize, p, alpha, TauPlus, TauPlus, param_hetero,amplitude), time_int,talbot_param); 
            dens_fpt_t(index) = Propag_dens_fun_t(2);            
            index=index+1;
            end             
        end   
    end

    function dens_fpt_t_hetero= dens_fpt_hetero_point(Nsize, x_h, x_start,ix_h)
        dens_fpt_t_hetero =zeros(t_points, 1);
        index=1;
        for time = timelog %logspace(1,6, t_points) 
            time_int = time:(time+1);
            alpha = 0.5; %parameter for homogeneous or heterogeneous cases
            xfinish =1; %xfinish = absorbing x %            xstart = 50; %or 100
            Propag_dens_fun_t = talbot_inversion(@(svalue) (svalue*propag_s_x_start_x_fin_heterogen(svalue, x_start, xfinish, Nsize, p, alpha, tauplus, tauplus, x_h, ix_h, tauplus,tauminus)),time_int,64);  % propag_s_x_start_x_fin_abs(svalue, xstart, xfinish, Nsize, p, alpha, TauPlus, TauMinus) ),time_int,64); % density for heterogeneous case 
            dens_fpt_t_hetero(index) = Propag_dens_fun_t(2);            
            index=index+1;
        end 
    end    

    function dens_fpt_t_het_avoid= dens_fpt_het_avoid_point(Nsize, x_h, x_start,xabs, alpha)
        dens_fpt_t_het_avoid =zeros(t_points, 1); % FPT for chain with avoiding link
        index=1;
        for time = timelog %logspace(1,6, t_points) 
            time_int = time:(time+1); %            alpha = 0.5; %parameter for homogeneous or heterogeneous cases
            xfinish =1; %xfinish = absorbing x %            xstart = 50; %or 100
            propag_x_0_x_avoid = talbot_inversion(@(svalue) (svalue* propag_s_x_start_x_fin_Qmatr(svalue, x_start, xabs, Nsize, p, alpha, tauplus, tauplus, x_h)),time_int,64);
            %Propag_dens_fun_t = talbot_inversion(@(svalue) (svalue*propag_s_x_start_x_fin_heterogen(svalue, x_start, xfinish, Nsize, p, alpha, tauplus, tauplus, x_h, ix_h, tauplus,tauminus)),time_int,64);  % propag_s_x_start_x_fin_abs(svalue, xstart, xfinish, Nsize, p, alpha, TauPlus, TauMinus) ),time_int,64); % density for heterogeneous case 
            dens_fpt_t_het_avoid(index) = propag_x_0_x_avoid(2);            
            index=index+1;
        end 
    end    




%%  plotting FPTD(t) for different
x_start=50; %exceptional change of x_start
dens_theor = dens_contin(Nsize);
dens_fpt_t_homotauplus = density_fpt(Nsize, tauplus, tauplus, x_start);  %dens_fpt_s = density_fpt_s(Nsize);
dens_fpt_t_homotauminus = density_fpt(Nsize, tauminus, tauminus, x_start);  %dens_fpt_s = density_fpt_s(Nsize);
dens_x0x_t_homo_dif_tau = density_fpt(Nsize, tauplus, tauminus, x_start); 
dens_x0x_t_homo_dif_tau2 = density_fpt(Nsize, tauminus, tauplus, x_start);

f= figure; 
set(f,'name','FPTD for homogeneous distribution')
plot_function(timelog, interval, dens_theor,'-.'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_fpt_t_homotauplus,'ro'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_fpt_t_homotauminus,'*:'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_x0x_t_homo_dif_tau, 's:'); %plot_function(slog, [1:t_points], dens_fpt_s,'b*'); %plot density of fpt(s)
plot_function(timelog, interval, dens_x0x_t_homo_dif_tau2, 'd:'); 
legend('FPTD(t) continuous', 'FPTD(t), \tau=1','FPTD(t), \tau=10', 'FPTD(t), \tau_+=1, \tau_-=10', 'FPTD(t), \tau_+=10, \tau_-=1') %legend('Density of FPT(s)')
hold off;

%% Plotting FPTD for heterogeneous tau distribution
x_start = 100;
amplitude = 50; %sinus amplitude for param_hetero=0
dens_fpt_t_homotauplus =density_fpt(Nsize, tauplus, tauplus, x_start); 
param_hetero=1; %tau(x) =a*x dependence if param_hetero=1
dens_fpt_t_hetero1 = density_fpt_hetero(Nsize, tauplus, tauminus, param_hetero,x_start, amplitude);
param_hetero=0; % tau(x) = sin(x) dependence if param_hetero=1
dens_fpt_t_hetero2 = density_fpt_hetero(Nsize, tauplus, tauminus, param_hetero,x_start, amplitude);
param_hetero=2; % 1/x dependence if param_hetero=0
dens_fpt_t_hetero3 = density_fpt_hetero(Nsize, tauplus, tauminus, param_hetero,x_start, amplitude);

f= figure; 
set(f,'name','FPTD for heterogeneous tau distribution')
plot_function(timelog, interval, dens_fpt_t_homotauplus,'r-'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_fpt_t_hetero1, 'bo:'); %plot_function(slog, [1:t_points], dens_fpt_s,'b*'); %plot density of fpt(s)
plot_function(timelog, interval, dens_fpt_t_hetero2, 'k*:'); %plot_function(slog, [1:t_points], dens_fpt_s,'b*'); %plot density of fpt(s)
plot_function(timelog, interval, dens_fpt_t_hetero3, 's:'); %plot_function(slog, [1:t_points], dens_fpt_s,'b*'); %plot density of fpt(s)
legend('FPTD(t), \tau=1, x_0=100', 'FPTD(t), \tau(x) =ax', 'FPTD(t), \tau(x) =0.5sin(\pi x/25))+1', 'FPTD(t), \tau(x)=b/x');
hold off;

%% Plotting FPTD for heterogeneous nodes at x=25, x=75, xstart =50
xstart_hetero =50;
ix_h=0; %distance between two heterogeneous nodes
x_start = 50;
amplitude = 50; %sinus amplitude for param_hetero=0
dens_fpt_t_homotauplus =density_fpt(Nsize, tauplus, tauplus, x_start);
dens_fpt_t_hetero_point_x25= dens_fpt_hetero_point(Nsize,25, xstart_hetero,ix_h);
dens_fpt_t_hetero_point_x75= dens_fpt_hetero_point(Nsize,75,  xstart_hetero,ix_h);
xstart_hetero =50;
ix_h=50; 
dens_fpt_t_hetero_point_x25_2hetero= dens_fpt_hetero_point(Nsize,25, xstart_hetero,ix_h);
dens_fpt_t_hetero_avoid_link25 = dens_fpt_het_avoid_point(Nsize, 25, xstart,xfinish, alphahetero);
y =timelog.^(-1-alphahetero); % fiting power law
% t = reshape(timelog,t_points,1);
% fitcurve = fit(t,dens_fpt_t_hetero_point_x25,'b*x^m'); 
%plot(fitcurve,'k')

f= figure; 
set(f,'name','FPTD for tau heterogeneous pointwise')
plot_function(timelog, interval, dens_fpt_t_homotauplus,'r*-'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_fpt_t_hetero_point_x25,'o:'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_fpt_t_hetero_point_x75, 's:');
plot_function(timelog, interval, dens_fpt_t_hetero_point_x25_2hetero,'d:'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_fpt_t_hetero_avoid_link25,'b:'); %plot(timelog, dens_fpt_t(time_index))
plot(timelog,y)

hold on;
legend('FPTD(t) \tau =1', 'FPTD(t) at x_h=25, x_0=100','FPTD(t) at x_h=75, x_0=100', 'FPTD(t) at x_h=25, x_h=75',  'FPTD(t) at x_h=25, x_0=100, avoiding');
hold off;


%% Plotting FPTD for heterogeneour tau at x=25, 75 with xstart=100;
xstart_hetero =100;
ix_h=0; %distance between two heterogeneous nodes
dens_fpt_t_homotauplus = density_fpt(Nsize, tauplus, tauplus, xstart_hetero);  %dens_fpt_s = density_fpt_s(Nsize);
dens_fpt_t_hetero_point_x25_100= dens_fpt_hetero_point(Nsize,25, xstart_hetero, ix_h);
dens_fpt_t_hetero_point_x75_100= dens_fpt_hetero_point(Nsize,75,  xstart_hetero, ix_h);

f= figure; 
set(f,'name','FPTD for tau heterogeneous pointwise')
plot_function(timelog, interval, dens_fpt_t_homotauplus,'r*-'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_fpt_t_hetero_point_x25_100,'bo:'); %plot(timelog, dens_fpt_t(time_index))
plot_function(timelog, interval, dens_fpt_t_hetero_point_x75_100, 'd:');
hold off;
legend('FPTD(t) \tau =1', 'FPTD(t) at x_h=25','FPTD(t) at x_h=75', num2str( xstart_hetero));
hold off;


end
