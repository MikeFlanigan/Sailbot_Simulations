
% aero curves
aoa = linspace(5,50,50);
cl = zeros(1,50);
cl(11:50) = 0.025*aoa(11:50);
m = cl(11)/(aoa(11)-aoa(1));
b = -m*aoa(1);
cl(1:10) = m*aoa(1:10)+b;

cd = .001*(aoa).^2;
clcd = cl./cd;

figure(1)
plot(aoa,cl,'g',aoa,cd,'r',aoa,clcd,'k')
title('aero curves')

% foil curves
aoa = linspace(0,50,50);
cl = zeros(1,50);
cl(16:50) = 0.025*aoa(16:50)+.35;
m = cl(16)/(aoa(16)-aoa(1));
b = -m*aoa(1);
cl(1:15) = m*aoa(1:15)+b;

cd = .001*(aoa).^2+.1;
clcd = cl./cd;

figure(2)
plot(aoa,cl,'go',aoa,cd,'r',aoa,clcd,'k')
title('hydro curves')