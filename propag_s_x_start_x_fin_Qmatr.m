%function returns a number Px0x(s) for each svalue, x0 = xstart, x=xfinish
%for given Qmatr

function propag_x_0_x = propag_s_x_start_x_fin_Qmatr(svalue, xstart, xfinish, Nsize, p, alpha, tauplus, tauminus, x_h)  
%alpha is parameter of Mittag Leffler distribution
% x_h is heterogeneous node and is set below N_pert = 2; %add N_pert links to the interval

%% Construction of Qmatr
Seqzeros = zeros(1,Nsize-2); 
toep_col = cat(2,0,p,Seqzeros);
toep_row = cat(2,0,1-p,Seqzeros);
Qmatr = toeplitz(toep_col,toep_row);
Qmatr(1,1) = 1; %absorbing boundary
Qmatr(1,2) = 0; %absorbing boundary
Qmatr(Nsize,Nsize) = 0; %reflecting boundary
Qmatr(Nsize, Nsize-1) = 1; %reflecting boundary


Qmatr_s_pert = Qmatr*1./(svalue*tauplus+1);
Qmatr_s_pert(x_h, x_h-1) =p/(1+svalue^alpha*tauplus^alpha);
Qmatr_s_pert(x_h-1, x_h) = (1-p)/(1+svalue^alpha*tauplus^alpha);%(1-p)/(1+svalue*tauminus);%


%% Qmatr_s_pert with link avoiding heterogeneous node
Qmatr_s_avoid_link = Qmatr_s_pert;
Qmatr_s_avoid_link(x_h-1, x_h) = 1./3*1./(1+svalue*tauplus); %renormalised Qmatr_s_avoid_link
Qmatr_s_avoid_link(x_h-1, x_h+1) = 1./3*1./(1+svalue*tauplus);
Qmatr_s_avoid_link(x_h-1, x_h-2) = 1./3*1./(1+svalue*tauplus);
Qmatr_s_avoid_link(x_h+1, x_h) = 1./3*1./(1+svalue*tauplus); %renormalised Qmatr_s_avoid_link
Qmatr_s_avoid_link(x_h+1, x_h-1) = 1./3*1./(1+svalue*tauplus);
Qmatr_s_avoid_link(x_h+1, x_h+2) = 1./3*1./(1+svalue*tauplus);

%% calculate Px0x(s) propagator  for perturbed matrix with heterogeneous node
sum_Qmatrs = sum(Qmatr_s_avoid_link, 2);
sum_Qmatrs_xind_pert = sum_Qmatrs(xfinish);
Inv_matr = inv(eye(Nsize, 'double')- Qmatr_s_avoid_link); %Inv_matr_xstart_xind  = Inv_matr(xstart, xind);
propag_x_0_x  =(1-sum_Qmatrs_xind_pert)/svalue * Inv_matr(xstart, xfinish);

end