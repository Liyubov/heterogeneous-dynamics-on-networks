%function returns a number Px0x(s) for each svalue, x0 = xstart, x=xfinish

function propag_x_0_x = propag_s_x_start_x_fin_heterogen(svalue, xstart, xfinish, Nsize, p, alpha, tauplus, tauminus, x_h, ix_h, tauplus_h, tauminus_h)  
%alpha is parameter of Mittag Leffler distribution
% x_h is heterogeneous node and is set below N_pert = 2; %add N_pert links to the interval


%% Qmatr_s_pert generation
Seqzeros = zeros(1,Nsize-2);
psi1 =1/(svalue*tauplus +1); %1/(1+svalue^alpha*tauplus^alpha);%  Cauchy distribution: b/(b^2 + t^2) 
psi2 = 1/(svalue*tauminus +1); %1/(1+svalue^alpha*tauminus^alpha); %
toep_col_s = cat(2,0,p*psi1,Seqzeros);
toep_row_s = cat(2,0,(1-p)*psi2,Seqzeros); %careful with (1-p) vs. p
Qmatr_s_pert= toeplitz(toep_col_s,toep_row_s); %disp('matrix Q(s) is calculated in funPropagfuncsx')%disp(size(Qmatrs))
Qmatr_s_pert(1,1) = 1/(svalue*tauplus +1); %absorbing boundary
Qmatr_s_pert(1,2) = 0; %absorbing boundary
Qmatr_s_pert(Nsize,Nsize) = p/(1+svalue*tauplus); %reflecting boundary (1-p)/(svalue*tauminus +1)
Qmatr_s_pert(Nsize, Nsize-1) =  (1-p)/(1+svalue*tauminus); %(1-p)/(1+ svalue*tauminus); %reflecting boundary with tauminus!

%% DISTRIBUTIONAL perturbations of Qmatr for interval
Qmatr_s_pert(x_h, x_h-1) =p/(1+svalue^alpha*tauminus_h^alpha);%p/(1+svalue*tauplus); %
Qmatr_s_pert(x_h, x_h+1) = (1-p)/(1+svalue^alpha*tauplus_h^alpha);%(1-p)/(1+svalue*tauminus);%
Qmatr_s_pert(x_h +ix_h, x_h+ix_h-1) = p/(1+svalue^alpha*tauplus^alpha); %p/(1+svalue*tauplus); %Mittag-Leffler distribution for a node
Qmatr_s_pert(x_h +ix_h, x_h+ix_h+1) = (1-p)/(1+svalue^alpha*tauminus^alpha);%(1-p)/(1+svalue*tauminus); %Mittag-Leffler distribution for a node    




%% calculate Px0x(s) propagator % funploteigs(Qmatr_s_pert) %plot eigenvalues for perturbed (heterogeneous)
sum_Qmatrs = sum(Qmatr_s_pert, 2);
sum_Qmatrs_xind_pert = sum_Qmatrs(xfinish);
Inv_matr = inv(eye(Nsize, 'double')- Qmatr_s_pert); %Inv_matr_xstart_xind  = Inv_matr(xstart, xind);
propag_x_0_x  =(1-sum_Qmatrs_xind_pert)/svalue * Inv_matr(xstart, xfinish);

end
