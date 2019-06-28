%calculate propagator P_x0x(s), for absorbing node x=0, reflecting x=N, 

function propag_x_0_x = propag_s_x_start_x_fin_abs(svalue, xstart, xfinish, Nsize, p, alpha, tauplus, tauminus) %alpha is used 

Seqzeros = zeros(1,Nsize-2);
%% Test construction of Qmatr
function Qmatr_pert = Qmatr_fun(Nsize, p)
    Seqzeros = zeros(1,Nsize-2); 
    toep_col = cat(2,0,p,Seqzeros);
    toep_row = cat(2,0,1-p,Seqzeros);
    Qmatr_pert = toeplitz(toep_col,toep_row);
    Qmatr_pert(1,1) = 1; %absorbing boundary
    Qmatr_pert(1,2) = 0; %absorbing boundary
    Qmatr_pert(Nsize,Nsize) = 0; %reflecting boundary
    Qmatr_pert(Nsize-1, Nsize) = 1; %reflecting boundary
end


%% Matrix Q(s) is calculated here for given svalue as Qmatr_s_pert
psi1 = 1/(1+svalue*tauplus);%1/(svalue^alpha*tauplus^alpha +1); 
psi2 = 1/(1+svalue*tauminus); %1/(svalue^alpha*tauminus^alpha +1);
toep_col_s = cat(2,0,p*psi1,Seqzeros);
toep_row_s = cat(2,0,(1-p)*psi2,Seqzeros); %careful with (1-p) vs. p % Qmatr_s_pert = Qmatr_pert/(1+ svalue*tauplus); %for tau+=tau-
Qmatr_s_pert= toeplitz(toep_col_s,toep_row_s); %disp('matrix Q(s) is calculated in funPropagfuncsx')%disp(size(Qmatrs))
Qmatr_s_pert(1,1) = 1/(svalue*tauplus +1); %absorbing boundary
Qmatr_s_pert(1,2) = 0; %absorbing boundary
Qmatr_s_pert(Nsize, Nsize) = p/(1+svalue*tauplus); %reflecting boundary : 0 or prop.to p
Qmatr_s_pert(Nsize, Nsize-1) =  (1-p)/(1+svalue*tauminus); %(1-p)/(1+ svalue*tauminus); %reflecting boundary with tauminus!


%% calculate propagator  P_x0x(s) by analytic formula
sum_Qmatrs = sum(Qmatr_s_pert, 2);
sum_Qmatrs_xind_pert = sum_Qmatrs(xfinish);
Inv_matr = inv(eye(Nsize, 'double')- Qmatr_s_pert); %inverse or pseudo-inverse pinv
propag_x_0_x  =(1-sum_Qmatrs_xind_pert)/svalue * Inv_matr(xstart, xfinish);

end
