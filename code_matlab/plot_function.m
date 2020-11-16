%plotting function for function on discrete x_axis

function plot_function(x_axis, x_int, function_x, style ) %,legend1,index) %xlabel, ylabel


plot(x_axis, function_x(x_int), style); %plot function_x(x_int) for indices

%f= figure; %set(f,'name','P_{t,exact}(x_0|x), P_{t,appr}(x_0|x), P_{t,num.inv} for circular graph ','numbertitle','off')%color={'red','blue',[0.7,0.2,0.5]};%%Define color either with names or in RGB %'color',color{1,part},
%set(hc, 'LineStyle', 'none') %without lines

%% legend1 = ['FPTD' ]; %insert inside the main program
% legendInfo(index) = legend1;
% legend(legendInfo); %,legend2, legend3);
set(gca,'fontsize',16) 
hold on;

xlabel(' t ') 
ylabel(' \rho_{x_0x}(t)')

xlim([10^2 10^6])
ylim([10^(-7) 10^(-3)])

%% loglog
set(gca, 'XScale', 'log') %log scale for y 
set(gca, 'YScale', 'log') %log scale for y  % %plot(t,Prop_t_num_inv,'go'); %plot analytic formula and numerical inverse on the same plot


%%
%filename = ['densities_for_different_heterogeneities_interv_N_' num2str(Nsize) '_q_' num2str(p) '_time_max_' num2str(time_max) '.jpg'];%saveas(fig_prop,filename)
%grid on;

% set(hl, 'Location', 'NorthEast');
% set(hl, 'Box', 'off');


end