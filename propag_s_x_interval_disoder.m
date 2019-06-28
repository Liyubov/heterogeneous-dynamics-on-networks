
%% function to calculate propagator Px0x(s) on disodered media of types:
%1. smooth function i.e. space dependent diffusion coefficient D(x)=a^2/2f(x)
%2. random distribution with parameters lambdaplus, lambdaminus
%tau_period_index is type of heterogeneity for the interval

function propag_x_0_x =  propag_s_x_interval_disoder(svalue, xstart, xfinish, Nsize, p, alpha, tauplus, tauminus,tau_period_index,amplitude) 
 
%% Q(s) matrix on an interval with reflectin and absorbing boundaries
Qmatr_s_pert= zeros(Nsize,Nsize);
Qmatr_s_pert(1,1) = 1/(svalue*tauplus +1); %absorbing boundary
Qmatr_s_pert(1,2) = 0; %absorbing boundary
Qmatr_s_pert(Nsize,Nsize) = p/(1+svalue*tauplus); %reflecting boundary (1-p)/(svalue*tauminus +1)
Qmatr_s_pert(Nsize, Nsize-1) =  (1-p)/(1+svalue*tauminus); %reflecting boundary with tauminus!

%% 1. DISTRIBUTIONAL perturbations of Qmatr for interval
for x_ind=2:Nsize-1
    if tau_period_index ==1
        a_tau = 2/(Nsize+1);
        tau_x_plus = a_tau*x_ind; %sin(x_ind*pi/100)*0.5*x_ind ;% %choose function f(x): tau_x prop f(x) 
        tau_x_minus = tau_x_plus; % tauminus;
    elseif tau_period_index ==0
        tau_x_minus = 0.5*sin(x_ind*pi/amplitude) + 1;%
        tau_x_plus =tau_x_minus;
    else 
        tau_x_plus =   Nsize/(5.1874*x_ind);  % 5.1874 =sum(1/x)   %1-(1/Nsize)*x_ind;%1/x_ind;% %normalised by sum(1/i)=5.7 ;% %choose function f(x): tau_x prop f(x) 
        tau_x_minus = tau_x_plus; % tauminus;         %atau_sq =Nsize/338350; % atau_sq for  atau_sq*x_ind^2;%
    end    
    Qmatr_s_pert(x_ind, x_ind-1) =p/(1+svalue^alpha*tau_x_minus^alpha);%p/(1+svalue*tauplus); %
    Qmatr_s_pert(x_ind, x_ind+1) = (1-p)/(1+svalue^alpha*tau_x_plus^alpha);%(1-p)/(1+svalue*tauminus);%
end

%% 2. DISTRIBUTIONAL perturbations for Qmatr with DISODER from PDF
%tau_x_min = exprnd(tauplus,1, Nsize); %smooth(tau_x_min) %normrnd(lambdaplus, 1, Nsize, 1); %poissrnd(lambdaplus, 1, Nsize); %function tau(x) from Poisson distribution
% polynomialOrder = 2; % windowWidth = 21; % tau_x_min = sgolayfilt(tau_x_min, polynomialOrder, windowWidth);
% tau_x_plus = poissrnd(tauminus,1, Nsize); % tau_filter = filter(, tau_x_min,x);
% x = [1:Nsize];
% noiseAmplitude =0.01;
% tau_x_min = sin(x) + noiseAmplitude * rand(1, Nsize);
% for x_ind=2:Nsize-1
%     Qmatr_s_pert(x_ind, x_ind-1) =p/(1+svalue*(tau_x_min(x_ind))); 
%     Qmatr_s_pert(x_ind, x_ind+1) = (1-p)/(1+svalue*(tau_x_min(x_ind)));
% end

Qmatr_s_pert(1,1) = 1/(svalue*tauplus +1); %absorbing boundary
Qmatr_s_pert(1,2) = 0; %absorbing boundary
Qmatr_s_pert(Nsize,Nsize) = p/(1+svalue*tauplus); %reflecting boundary (1-p)/(svalue*tauminus +1)
Qmatr_s_pert(Nsize, Nsize-1) =  (1-p)/(1+svalue*tauminus); %reflecting boundary with tauminus!


%% 3. Pointwise REGULAR disorder in even x points
% for x_ind=2:2:8
%    Qmatr_s_pert(x_ind, x_ind-1) = p/(1+svalue*tauminus);
%    Qmatr_s_pert(x_ind, x_ind+1) = (1-p)/(1+svalue*tauminus);
% end

%% calculate propagator for Q(s) matrix
sum_Qmatrs = sum(Qmatr_s_pert, 2);
sum_Qmatrs_xind_pert = sum_Qmatrs(xfinish);
Inv_matr = inv(eye(Nsize, 'double')- Qmatr_s_pert); %Inv_matr_xstart_xind  = Inv_matr(xstart, xind);
propag_x_0_x  =(1-sum_Qmatrs_xind_pert)/svalue * Inv_matr(xstart, xfinish);


end 
